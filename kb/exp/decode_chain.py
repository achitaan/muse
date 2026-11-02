# okc: problem=muse_decode_layer
# stack_all1: stack_mf1 (MFIN, no MoE grid barrier) + SC6 (6-bit block scales, WGS MoE builds) + PRF32 + PDL at T=128 (moe-fin bundle; L2 hints at T=1 left out: judge d1 25.7-26.6 us with them vs 24.5-25.8 without).
# st2l: st1l with OKC_WGSFCI 4 (E374).
# st1l: ybfl.py + OKC_WGSFCI 5 + weight-ring pacing at M <= 64 (E372).
# ybfl: cepil_u160.py + WGS ybuf in bf16 (epilogue store + final-pass reads; harness/make_ybf.py).
# cepil_u160: pcx_h1 + in-layer MoE per-unit stream-K cost U = 160 (A) / 140 (B), zhebrak SK_CEPI (harness/make_cepi.py --layer).
# pcx_h1: pre-wait control of the sk2 kernels: hoist (harness/make_pcx.py; team keell0renz; base stack_all1_snap1)
# + alygg77/okc-muse-spark-kernels kernels/muse_decode_layer.py (Apache-2.0, commit 6cae49a); header below is theirs. Patched by team keell0renz: trig, rq, atrig0, rtrig8, pf4.
# bt_sc6: lossless 6-bit block scales in the layer's WGS MoE builds (T >= 17), moe-scale SC6 via harness/make_bt_sc6.py.
# stack_df1: data-flag A->B hand-off in the MoE MID build (T = 9..16), harness/make_dl_df.py (moe-small E173).
# + chenwainuo/okc muse_bf16_proj rs128 GEMVs (Apache-2.0) at T = 128 / 256 for qkv and T = 128 for the unpooler (team keell0renz).
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn.

Pipeline (one decode step, T = batch * q_len tokens)
-----------------------------------------------------
1.  qkv projection (cuBLAS, the bf16_proj kernel, or a Triton split-K GEMM --
    prepare() times the three and pins the winner per (T, matrix)).
2.  Attention: the muse_attention_decode entry's fused kernel (QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head
    RMSNorm * sigmoid(gate)) for the shapes it supports; otherwise the glue's
    own QK-prep Triton kernel plus SGLang's FA3 (or a Triton split-KV
    attention fallback when SGLang is unavailable).
3.  o projection, then the first residual boundary (residual add, residual
    gate, pre-FFN norm) -- one-program-per-token kernels or parallel
    atomic-sum kernels, per T, timed in prepare().
4.  Pooler + router (bf16_proj's fused launch, its two separate launches, or
    the Triton split-K GEMM), routing (top-8 of biased sigmoid scores), and
    the pre-expert norm.
5.  Experts: the muse_moe_w4a16 entry's NVFP4 kernel (three compile-time
    builds by token count), or SGLang's Marlin NVFP4 path, whichever
    prepare() timed faster per T.
6.  Unpooler, then the second residual boundary (post-FFN norm, next layer's
    pre-attention norm).

Provenance: the attention, MoE and dense kernels are our own team's entries
on this challenge's component boards, copied (not imported). The CUDA sources
(ATT7_CUDA_SRC, MOE_CUDA_SRC, the bfp_* template strings) and the bfp_*
generators are kept verbatim -- they are the tuned, proven kernels. The glue
(boundaries, routing, tuning, launchers) is this file's own code.

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding
with exact escape correction); NVFP4 experts stay NVFP4; norms apply
(1 + w) formed in FP32 at use; FP8 queries / FP8 KV cache as SGLang;
softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too;
FP32 accumulation throughout; activations BF16 wherever SGLang keeps BF16.

Selection: prepare() times each backend per (token count, matrix) and the
expert kernels per token count, on the judge's own GPU with synthetic
activations (it reads only weights, never data), then fixes the choices;
untuned token counts take the next tuned bucket's.

OKC rules: one Python file; native CUDA only via
torch.utils.cpp_extension.load_inline; CUDA-graph capturable (no host syncs
in timed paths -- prepare() and the tuners sync, kernel_fn does not); fresh
inputs each call, no caching across calls (module-level caches hold only
compiled extensions and per-device scratch, never activations).
Safety: every scratch/workspace buffer is followed by sentinel words;
prepare() checks them after the tuner and the attention self-test.
"""

import os
import threading
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# Model shape (the task's dimensions, not tunables). The CUDA sources carry
# their own copies because they compile standalone; keep them in sync by hand.
HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV_ROWS = 2304
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024: q / o-projection width
ROPE_THETA = 500000.0
ROUTER_MULT = 0.1767766952966369  # router logit scale (1 / sqrt(32))
LOG2E = 1.4426950408889634

_TUNE_LOG = False
FORCE_PBMAX = None  # override the tuner's pbmax (no consumer; key kept for state shape)
# Programmatic Dependent Launch for this file's Triton glue: each kernel launches
# while its producer drains and waits (griddepcontrol.wait) right before its
# first read of producer data; weights and older inputs load before the wait.
PDL = True
PDL_MAX_T = 128  # a launch-latency win at decode sizes, a ~1% loss at T = 256


def _pdl(T):
    return PDL and T <= PDL_MAX_T


# The MoE agent's kernel wins at every token count we tune; the tuner may only
# depart from it when another backend is clearly faster, so a noisy measurement
# cannot flip a pick.
DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
# Dense picks pinned to the winners measured in the whole layer: the isolated
# tuner sometimes gets them wrong because it cannot see PDL overlap in the chain.
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
FORCE_GLUEV2 = None  # None: tuner's choice; True/False: force the parallel boundary/router kernels


# ---------------------------------------------------------------------------
# Attention: fused decode kernel (verbatim CUDA source, own extension).
# QK-norm, RoPE, FP8 cache write, split-KV FP8 attention, merge,
# per-head RMSNorm * sigmoid(gate).
# ---------------------------------------------------------------------------
# achitaan CH: ATP (8ebcc1e) source-storage codec; decodes CUDA text only.
import base64 as _ch_codec_base64
import zlib as _ch_codec_zlib


def _ch_source(blob):
    return _ch_codec_zlib.decompress(_ch_codec_base64.b85decode(blob)).decode("utf-8")


# achitaan CH: lossless CUDA storage, bytes=191952 sha256=575c21f8950cd19a379abd1d40f8f06c10f764b27b101c2ab417ac397ebc1b07
ATT7_CUDA_SRC = _ch_source(
    'c-ri}dwbhPk}v$fJ_X`^>;M)?^JbYLE$7I#?D(-PYb3><S+BzZNl0SEn*dUmB+h4l>sQs?=mtQFay)xx-?P~ri@5cry1KgVwecUr'
    'Nq@8$MBdZ+H13~0imvC;BpFU8m9u9X>BF7lX!5AP7=({r{QPp~#dI>K7u{FpH?ycakHg_Sv2Wv}?#ZmS<-X0gdnbjrPiC$BUH48h'
    '9}FivfQ?Bwj*?l}kG$VEHvVG}oeU?D_x|mR?*4&StG7J<@6jV~7EcF@eiVE2vp5O|iRWK*N72Nqc_*WA-r;u>>LBtiYqeV2qp^AA'
    'YURoOl6TUm<7e}}cbRzg$6JBZ&&MNb!^`(`6~?pEc)FN*CsXtqaYvs2+<WHLsL{n}<c-3cC=R^yD4s;4j#u@zsrTyliM%SeGo7_e'
    'Z`sZg(~-=qWw+wZr_&60S_VGUlKQ0WHM5QSw4z>rI!PAeC{Dad6b-nOI656BbGnrU$GbQC&%L^H^Jw?@-Ok_l4|iXCHRr(r-8%BB'
    '&OLhd;I*2iJ;(8M5VgHocp7=b$za%z60djT(c7LsrBD0Q*^M{D0uM(~K(El?!httE!OgO)x;L8A(mTEV{imJ1R|g*m_S~CK-Sy~m'
    'g&b|~EF7KieCeI5X?$KHNP0;Wc~@r<-Bth>ETZP8GjEuJtK6F|CIh#>{r4Yt-W+H9!x-b}Vi6_MO52-EX#s|Fv#6NX_h@=Hq?!At'
    'Vca871SNjhpN?mXc_jVutMtz}a(aGu`1<AZ*W5ApJcxRWQ?Iuejs^)0;4DlcZyd(wi5JGw3^h!XI7PQ75w?ckpN<yeNkWSf^+z@D'
    'k=Gy9-SvOBd-!^{F3U%~jKXAIesR1b4JB<4OBMDnx-sEoIG>J(eJ?zjD>&lCUKqzix-MQo9!HYzv0-}b6ZE1uoDVOf!cKMj_(QM!'
    '#m?Wn{QIjnheyZg9>{7fqij6t{q3U<gcQkwHqI1KL}v~fk3xdBKbg+0(Oh`(^eVx&&A@l(^_v$j-+U<Ouz$9goD;@Q>GwE1CCH8!'
    'qxo<)%FO6wFud#z-3|JpX44hG(4&XpX!slLIl|?x!n>>C{A{|IE2N_x$0$#GVd72t-u8B8fB)@($Y{;WP#N!UFW(&QzBr~y&ccKs'
    'bTVC}gc*C7=I;6Q>0mfcyeRCSxj=Zozq`Nt(JVP#Dm-rUz!IE8r)Tr>2p`Z68%<9ak={DtD49CDiXb+k^W>Ch+Z97S1De(f(4dV7'
    'eKB#6gGl=MUWg3=p3zTjsdzYZ3E-Q3wV2CVPez0z1c=Ka&Se76a55oG3VYK_K_!i|pcYvu5q*8hjc1@l`<GBJoKr)7FAgXDv+_wC'
    '5gl3yI;Zu$o#Wl(H>PzP5%p&@f#HA-iZbsW|8hz+l8JMQ=GX^i+6U3K>Z9b*>hK0A12+hf=Vvsk@CMst?ozaa*Dv1hAEy*8Cg}Ii'
    'qW(Du@SFw_*zMz_P_Rm5BLdS#;euvQy^f}G&drAt?9{LyPpSJPIvq!od6GY&TBhGhrE=TR>-pm-Ocn%n@6Sze6)rv>9k*W5$!m_h'
    ';VfeK)4Yj(Vb3mU+iB&+`}YUdX0r*QMkyXAu{o!s$gI|BJOn#bZPU)U0-Vv8q1*aPcfbc9KFajneLbCc$+-IiHASaoNLy_}|K1RM'
    '<LP9&AnI{C9*5pZJRN(XH<&JZG>qQj<Rk)Mz;S)x4e2rzuV3!INZA41Rfa9#y?iw_CvVDr(HhbSXd!1I0V5jS2)lr_KO?lvEX1yy'
    'k>(~{9UULOdAaNP*7F~|t)|!ZYFk;TIHY0)R9+GHCBX0xKp~>$xKYBhqt|=k<dhEFS33=UYX<xuOmT|Mrh`YsNi)arurj;Jbbk2$'
    'm0lCI9{~e(u+v8;Aax3r#vyeI(oG9L7>1`4Iy30JP|QHCT}-TM*fDNMQ<C#2nsJjDEX@z&HQ&>B@An0AXyU;G&;QPh^^ZU7_5(cq'
    'PT^J#tgndLXiB3xq6uV}YUZf}XV-ySP2?mxI-&)uHp`lCO?g_kH+s=tL>zdk%Ql@?)N67U0$s<^cp4Kj{q5~9$7)sSOI}oB%4nBH'
    'S0FFgeTG%ePV49&`!6(L4YP;ij=4?74?4GpB=Ud>=d;Q$YBw{3*5S^_{EksLh`G?lr=jyc1Pm9G;k<nK?nvGDlm0X&vXk_~5hx)O'
    'YfRJa25SP6Q{yQx!5Pn(^Q0z0RLWNvIWLeyf3)D06-j8+$3Gn!vU<@XD)ie8;p%mp+a|Dkad_pa;ql7&)02}VntPibVbCKlN%|J4'
    'C7MpKYI~ctDo80X{A5PL9C)EPgJUm#dTTQ(t+pbAzulB+0G{e*4uUzIsH3ZYPMu)udIw<NgjECth>75@2!a#0pW~ko_pM!eA=F$t'
    '134nL!J=H@h(JFfs->|#x}Fj7qBANSsg3QEj1pa9+0b*z)05xumi$M(Zq_P|`yH>|*3-giSkm)48OAh8KB(Q}W$);nHFTfG62(KD'
    'j)6BGPKM*f*c;rCJWgPC_XrUk65M&y$#Dw-hA;uVFK98&8PMaScbP?bx%c*i>XLe-OY^<MRrexlOmJ=x&D7an!G<c)mVLy0-FfK}'
    'yL2!l$z`rsGr?@)npVJy#H2o*FkKea+d}pP^=cg;)npKkFbCSaeIlZX_h*%WD|-tcLdGYCImlr8;jV*i3JVC!$5;h|eZm+u4il0%'
    '>4#=b1Mtcd5hOaE$i1>NwCI~d>*L*{W3@vWeb1)TbKZvr@AW1c6HAE|p&tqZGL+@#{oN0lGsXxX;9P~3VCamoA_>VJAY5l4j_+vE'
    '78Al%JwcM*2SZ|+=S1(WLZWB>Rkef3p3>>GYE5W_y&6M}2!5NSk16pHyzx$nmtBm)csRc)d5={;fDDhM<k5bFn3bGO<9Reu{58oM'
    '=su&zsP5G=Qzi6Q<7f0v^(~X>q-=9rTD24i@e`=HT;OJ-QhDqxL||Dp*VAai%~BV{44y^<MfK&L!Z`*uLpkLWYsBo5U`cE)z?tOu'
    'oVP7+tE$c(9;Q(NT@K^<f(Xocbfcs(wZ^Pn2x&tPH3f|4Lmr0TgPd1=QfYc+I*F^q%=C!R)>`^btC=4_&1)76pr*v0a2!dqKUAfS'
    'R;%(PyGBPO;Ox9q*!{8U!bS^5gw-*fe=P9RwE8p)9GV3O1s#%@s-myc(I~6js_yO_?)^h^A{b}3Q_*Q88Pdeiim9P%2-DOxvt~7='
    'z2T$Z$VB5zm#^L*zSw<kW~q*#(GA4rSg8{-qkzID9F2N(LJ^(j(*ZPg9-c?S-U}9EPh=<fT#0nt7wx8V!&*}dEQ;`6#^&vfrv39b'
    'gKN*{qY@xwigX&y6E!cIWJ2V6uow}*zb6n3r`VhARTw{Hc43&{l#sKCz$%lV1Hx8h&=}OBI+Gv{{r!G=Y{)qYbr_NBaWTi+*31jF'
    'yO=AT8YIbtcNqOcRG-;~MIQimIl1RMdj~s*MyDsVT<t!5Hjhq0f`n+W^wFr7do)ii)jPB5KRkZ)IwRO=DL36L8Tm&}0v28&XtCg!'
    '%8?mimXDS}3>koFquh2e;pIQ}Z3(ZaPkq?4-iZJ;3JJyoE#94uroD8x*{nFGlVK&_?0?)je5V08prt0>bEKt}oG|JQ7BLhyvaOVk'
    '!nf9kKJN<xj?0T#PIj$Xhp$Wf9F;5W*vlcnX}ED~JCmqN+asAx!GuA1k}#9LgeTvB|MwR^)l%M8NL|U&8nLdD)xHU_O~mQ-iJqSt'
    'i48lD2r~51IHdjFlnz;<T?_C58A6tw@6o~D8zVklh$zES1}nD}r6T4inSy}HJPx4878bVynhe0;t$C_8i$rVb&F0ti(TgXKYmZm$'
    'Yk^=wJR{bS=@h0rKf_8ADxQp3kEU={NDA>3(SWtw#H^~mC(-p>Ys2SLSDp5^xBuzrC0Z=Ev=tqZ5iomlMPw){gN#y#{l%D=hWRv3'
    'L<<X41tJp~@T@x2%71QGy-Q;Wav}U?|3lrvC?f_8Czr&EgtX(@-q|@ZqF2eHM}N+SU0`>}n%}5g^?EV!qkY2iq~u+W23_LEYIv-E'
    'SG`||Vz|1!|9s`B8VGeoMGpwQr#>Kp&{5<47R6H<2w}B_yF5CEE@Zuhl0SH^P*D<_k-?$AA9QyP4|n$Vs8i}rsIk$tDD}8#)Dta7'
    'Fbyh4<JtTs)w}uHQjvr-CfdrxU8GvOQ4~%({4vq9GJvxRp&Si9^-y6&&m*tf#ScDwu0pn@G2diP5hfjRv`A>@=p}94&rNv${kxr`'
    'w}ehObcZD2oIcVLZYuilNleFiJmof%#b}PxL<xh@DB_pE#%JsXhdcYPcO6-mtv)y`a7YRQ@mU{=khGX2!&7K}HN&sKtS}84PCsUf'
    'VM_>;XhCFZl%WVDxc{{O#-a$udL@_;qkR%N%Fx3LgANS`O=~#e1BIzG_Ws!v)NUZGq0-iBNu=YIYn8|Mvwa`FIsS)Hc;`e2g(+=*'
    'BGR4_gRyQY_l7eJO1(KzK7BjOWbPmB)(w!C_mxDG1PW0?4=6{4L@4bNmNgt0me-s%nrAexC39@AK{V1jR3@2mPJ}w#FR+O4W?W3D'
    '^^xgngaewHv@(godD9Xd0))J@ddX<W_6x#e$P{jeJ1<|F9$^Jgw@fd2i<O8fCnRyE@)MoDN=bnQM`zGUf_xKxWHgcQU%hJ7(;gMC'
    'BPhxU!b+g?OAthJ{5_H*o7NTft1=S=9B4!$@(^r!^&A9K1FQn0UzZVo_~qCWB{LL6!WglHNEDIw67x+E<mWO|ShWaqn>oxq+<n<B'
    'Sh>szu5<%i<=)LaGSYz0gnkwdiQBpgXIW|N<IVxh&>2HGEG(=ctfiVr-a>Edv^+H3Ae|uL<@Jp^0$@!{y$`e$1mRxsX2$xGaxnAh'
    'DoXq7B5}@-1EqQ);UZ+=ok(p=1d2HaOGlO33$Yh~HJY`W!gfv)!H29?QBQhbP8tUehA*>lme60Lt<QUf26q>@+-2IrMbPAhZ%SGA'
    'SU8tFTr<pxF%>Vd6^n4d6rEVvC>*PqC~k*U%}OmSOk1{NL2os+(yF<cX$=!~ep3g~sVOTd!<vs4Z__(E-g&)y<dtdYkoD*YrCHGy'
    'lNZXh@0G+s{Os9~5MnaDoe>0IEA6w{H!y}jcpy!&4`ZHBOejUID!nq>oNYl&AmK_AER1FJ-!)qo<3HUdvZVDI2@7km{{?Nq<*Ih5'
    'NCZGc{b-6{zvYAlCMl`O8xcKRHCua$uss|#0G(mfv-U6Ok$6M=NfL!IOp*`<m|JE&GNfJgUbVy=!5SD$<eO%9S#Pz7)2VIIX@Viw'
    'Dzz4!6NCfvt0<b}v=gCn{;;R$?ow8RJl{D~q^U?cOgD^bMk7he3_|BfJ2ADC@^V9vlkU&L0r4Lq0Iiy367d=*tk!j8Y1`0jbXBab'
    '1xLK;_<RtfEht83!>|tp5S>*S=H>X{m30L8h9on}1sZt(%V^H0SrHWETJ0%0V&P|EdxQ`hIQ91GE4klPUdyf(M7Q(F^lBoS8NDz>'
    'Zsiab?e`o`8)pQDy^IUbtAoS$wxXXMg#xq4!v82;3N6*3cdD%C&1$*Xv?YBpmu{2zNE;T$pe;)ch;F(Bg?9)3c|h!h_q#Vd5n~~b'
    ';K2iLR-wiNdN2Zejej0`-NDS;mM64;zE(n2uIUQ9`{i&<zaLspZmcIa^u#>)EePl`HStn&=Rt=@@sPXO_Nw%T^mqyll24Dn(EkZR'
    '@~c740RQ^uox_)Mc7c&0sns#W`)n9fC>JPmh6oYtM1%DjGCP6;s}DUaJyKQ=61U;R@yI(m*nP2*xU*J^jwx{wA+7_+U|BwiSxy4>'
    'dr_<^ROk$(l1JxImig=uTYE`}dAqinT1fXd9fLEq1#4bn@5=;UEQ%ih*A1k>pU{KS1ENhhx$y?!oX&}u2;m?KHo9FIM7K*4o|Z)D'
    ';&MuTqQUM-I2`#vmo7X}XR=s=!{9-vUwk4q>qH;aGqdrb@EtOqO=s+BH?V91M+a~AyT{d34C`{26T%GES`oq#jRwQ774rmRtp0sb'
    'gaLbE5hcK7qA$_b4Ai-n$|H&QXYD@W-6Oa27eDR3c<b(y6ya#IY;NiyJ@fq9!h8DEt2YV5Ezopq)K#N?w(Zq_sW-L0j<w#9h=?nO'
    'U!XM#@l>h3@)};@*(yZ5opWFD8>xb-7hz98t<MgyKQy4(|Dncm3*2kBvW_wPy0@k7b(!v5S>fZ(;X%eH?G0SVs>|hvw$d`$(yuLu'
    'Yx&z(bbW#6fT_scmPYj!#T<)a#iC%F|1I|!AqMK))T-=9v>OvTrDm+LdSnYY$~zz&3}QO@J@|JRX?*1gD%yL+6dK4qBHVg?<oTCT'
    'pZc<Nn<WzM?<;?4e=!--XJhMeb`c%Jv7V=Ps=)ABb|wTEhV+wU#F$tSzb<1xGn+=KIt&$QSpfbMs)j-^qx*wHqj7+@)(Y9zCo&wf'
    'Irl1OH9LOs)6e^F3w0M_=~T8zI_An?79H`~<)8k3aPWTbA6ns^1<4s9-WU{+CMi0rrYlyFu%-1suiowO7N9L%sWfva*A@m@JcBb#'
    '+~0p{tymYQ!%!=O*-Z<AY~pwTWR%?y*Q#33u{G?bb#`yg%BV`iD{LU<9;&^S5fT6MAMbYlZq!$<e@mes#u;sokr-d-IMs3_tGO)O'
    'cY*dOH6uB$nMqz}l&Uz36jPS}r=De$U5}5Vr+-|nT4p3*&zNK_V+dfYtO+S#62#)Dddw%Vk2bd1%9mm7>P9`DrG4#;D@KV;4kCss'
    ';|`l#3{JoOz#c0FX4L5<DvE=UcIgZr(^kpAx_DfL^d7^qL}atUN|_1d2A1XJ4|4`CV+OL>W)%k^o*y7qX}e?4ep?Y}SZ&}i3|JGz'
    'e~DC`8^KFUk)>|6WpX!Q;*B0*U`aAcK-cm!xLea25L#3Nd2Mz&U!%~r6xqoVO?GlVoMr7@`L<u}?!Va8)ND!HR$Go%<`%69w@>sk'
    '-HA)vKOme}&SBVuHUZ1JGPz!!jDm2*=DYJ~JcF;X_mspiZ?EL_rqhx4>e+^99lRGOr{3=yw;`%=`*d9hmtn8C1SDRoQh^aBJQsb('
    'bTCkkIE>!XtA*p`avGr`RvjAnq<`b}&%nTII-%~W_z-)vtPr7Qm46)TLJ4rOOq%xPw&(9b5>g(Tukf$hvijFMnk4;ZZ_my~o~u&Q'
    'ed-6B*{Gs!x4m~2a5{1VC`to+-__Lik7KH|p^yS}N!gy2`<bG>MzeGhviQR~Hg<(zzZu^+H>lwdi_LoNczViQ6hR77Sg;gIud+}w'
    'KzdaSBz;qRwyI0Go@y^xw)q>E*yF=aQSZP{dQEp;3%nmWsH^SyKCRCmf6&ly?{Te8_Yk_)(TAC1x=Wdqlol2)T69gD)TN9<ZxY3)'
    'C=KF`@^DFJQ9m{J8Xez;GYABgUGk!bV*TjPD(p5cO?OhwX2JxHL`0Zp*{w%>&Ggy3bcH@15s+SG*FuHw!7W+fr+tz?9<_qy00)_&'
    '&weBzr(jnhhix5P7tqgYr#SX-`BMI!nx+Naq)s}^c=h~}6*k6UczU3OhlCkwbP}S60Rl<neqCdRy%C%+KRWCn1C8jk+fZ6J;?%Dv'
    '^n!4|ngbDKY5JWn+KaZg_4sjPi{urbhxQQDC5&Lkh?a8z6~%l?J80RAp6?v(a=akndr@0(0B+Im774)h$B$`d^(OCyl`k({=L6MO'
    '8mIV-&Y&Ny>$brF;1I3;(`{@&-f>&>v{7G<MKAPBd^0T`m{(RE)71e6=@ULk$MmC3$mUG5*;*>rS^pVGe0^t%uA8lCq;6t$;fHOD'
    'Jj6B(5EfugAFvcmRA3}3fjk3^JU<!J5A7PI-Gxii6Vql38%w}F?MDp!f?qs*kBh8AqUML9Nts4Tm>VFT?vVsz$!p8jMyX~&G4h+7'
    'H6U%ZrCMrBA%G`Hchr`-y&ovR85)47Cdd!_2aj43<Hu(l@2<Hva}YG4d=NLA>##ouYsQK>{^E2>GC{&(ht}1n#FXQ}gpMoan(dIG'
    '<&<=nbuplL?te$5g6N-5czmBG2IZ$h>t}SX;pbM9p7{F4k9hQmK|sJ^U1{FpbT1co%@gfUMcbz&dL=dsXzmfnzH~O+B{_bxw@VAM'
    ')yzIHYn6L)`2HgZAZ^IbGBxSVJW3TL04)(h&21#!HFa0iTle`tj=4)7)I;v5quXqGZ=Z_}h@lPSd`pN9Z&v*N5M)|>^YZWAqrLaX'
    'Xs=F<$$YYNA#oRTp>w>GpN1&rwSA*)%vxg3rd^Y7>}dD3K=andjL)DxZZsEs5XN9%!3cBnP|99*dnco5IInFzgn_?fo>8|CjhV)|'
    'z18f}Bewl_N7Fu@E$DU&ch9?g-??|0-m{v9@uWKsdn4-sj|7cy982#0<&yhDn#Bo$5Dzj_nl9$Jt6scHx?#WX-WcJMM?@F>$y{A}'
    '&r9A<7xcgL%aV6cLP$#&Lnc}%gbBysEGl`Et|*d9ybg5D?OWE5LGQGa`oYhzMo`)k>0^F6=jv)HpXscfEQG<DOk<+FOpVn5`w2Q5'
    'bsSvKK^tPw>nABW)=&JGNAGsui65fk@TirPN?YFJDV&mi45ezvQa_Hnt*RHX7ao-7M7X74pHPAZ3ia!d?E=DvGlYqdb#i*ld_zne'
    ')o@dbDds#HbT6<VwC}r23d~wc=o8`NX}x0_=6fCxTYh#=Le`2t^zfb^seyUK73H^;;-SGFGX=LZ7Z8OKZ;|9`_}g#rL1x9X@<nld'
    'gVph?S8Y0-<zSFM2j^!rC2-C-LG{6@!-j0XTKITLys-itM?))U%?d~l&zvon3CD=4SL0r%_*G0ys4k4N%1f8&E^Q5hB<zFI#`NT*'
    ';SxKgB4O{Mw#WXj%4Mh*ftzT0ub+Fq@-~u|MYop_Z_8(=I$80Y!xW{btLieE%FrtjS+>}aHoE~Z!=iyCN;tp~E`~VFp?s=U>rJDB'
    'lb$r_J1#pPNyxr(d1|L>ykTMve`gGQ=%@bCV_sX`IEbaTQB9IACj#hL@Y4wdXhTQCkzfC%QAdcBB0K>H0_!VbeAMU>`;tR89Tn6r'
    '4hF&qErx4VXb@&iQ)8K3^*9@hPI5gOTPU~1LV7qJGZ#m!XfzqH6<qA9ei}t)4A;|mAl%Q3pT*PUT{!dR$m?;b_WCDay+sy2dzZuL'
    '%7kIRg~>&sUJ0A`x+dGVH@#MOp6jig?PfObcuNwSn5+<djA>&Nfq~XX>@OuCFU^iNEqCSl?A%LSlciABr<pl0XD52~_v*$of#mQ7'
    'Vn@V3fF$H7CYkL@G8rh>BXwMCV;9*-H9P6D$A2fsMS(pa&ds&da*&EK)-HHL-8nnCcNdMiFT;fqm&O_}KM_asMLh9*{c_Ol_AlpQ'
    'mt&!bVVzIAA~E<g@QSPYW8-jt)@7<$+cHghg9PDVR^Lh7FEQ594_S8^Oqh&4ow>ljrwMSSe_D|-RFvWL)BWn_68&4Fe`{MM?|!}W'
    'Uz2-YyL~Ud=TmP4&wHbLKDCZtG5=Ei+;I|JWN*0BOq2=}-vwHAX{CS=h0SE@Nh*jy@<Xs?TPb&xn|>*6lAi+q%J2kBa)A2%zb0M<'
    'x`7I}bwR*@#$Z;NL=|>hB?vW2UKRKM&xLnKxPBqfW@U|aAT8HE0ViN!+zIl{)i7U;F5!x4HJ4(_y5mFO=y1-y56cF?Z**C4s=9+a'
    'Kt|cU|7@M!AYrcR3(ut|?|xO!ZF0}w54g*0E)B1`LHJmv|CGWU_rNJAKc!d<kHj*8B5GxyxKTK^e{SQ}37~;4+a9{`0`rU3jUQKZ'
    'SNt9!^C%+rx}um#HLtdJPt+Fo)><p>9F~jzygR>Ugc1pe?#rfKN_zBYDiZflF|8=fYPb9Iu+k$Ty<1Y~b-!j|mZ7-RPM~&z>g;PK'
    '0hn|rMKj<deB}%np4m~aIb_xP*9=(z=MH%tNH2l!(t&0NN7%iX^pu{{M3{v-g5NL#mR=q@?lK*`!Fn^}9S4t^qV*7O_o9gsk<QR)'
    '<R<Ih-){fc!~lW#4Q{_8?c|_w;b7bU`R)S8h^HD_Nvn&NR}VH1Le5*FH-8)v94=M<03F%O9(0B;wjG<JFy<^+Mt!M_sO%v(Vtc~c'
    '$3XW^xMYZ87OCLD)-T)ycfl+1)9Wop=W>dx)1~}O>r{OciZEjKt)!EJvG`Kk^7O$7x291fQZwA|46JVj40`W^R{L@mE@wy-GqOI='
    '&~yqTA-w7YH2|dtGW3XOVY@BdiG94@uj5yP;d%=q;`Gh24>s#?>w*wOnJbrJuph6R41;iTx;Q;cgWOY<I$~qMhN(r2IX%%G_(F8N'
    'f=3A_H{DzLl-t;^Y9X(po7PZpLhmut{o}mNQ5+7Cz|;7QlC;@T@q79$?H0?P4eZ^c$Yrh{kb>0P#BDh78Z0$8ZwDxLH}CTa_(Tsb'
    'vG_a5Gdw2d@P0LOa193vUnVY&cwg4)UP#1x15i<r-*O07r<fIYesx-eF3!ZEU{AYkc$@rDW@n^qrn1MDl*MxmhLZ;)!x)}erW*g_'
    'QLt=awNi%lJkAWs6P1vIYd#Nk)qq4h7+u$W`E@PBl8c*;0Ls5dFI2)Ayj&_L^WQ3nF;(kx52M&Vx#!CfM?bCu9*vTDZ*E~`%<i7W'
    '<9WK*!&3UBXZ=p1k&G~<3TBXKfkiPHn$XGn4K+k;ZyqB32sI(pM37pw2R+ay^wm_-A^AVOdHoX~F3|DNIYV%wBk82JbzN_(c<~eC'
    'e9jIm)Fu-1Fu{CPe|7LZyjP=`Gh<eMLZ%Gb`bR&#S_?S+%X$4Qh0gU-7!U*WUvpuB;FCP>;Cej0WVrlJeBS(v+E)KHr$#IBq{5S^'
    'V4BwgI{D~J9&FdQ00~OVVi?IYKDOx=egj6q+H?QwQ>UY&`)#Mb7E+U>*FDplYcwsXopm;pX>_}QL)|;`w$TWlC3U{ljVC_dr#G#R'
    'AI1>}R1<qU?;^hbsqX6GGylPvKCSA%;M4jqXF~~d&Um`ERaC}of&31NjVCn4bN`R)KVbH6c{rikUKGc~+CPO)eYqt*Ti&~$U;RwT'
    'ycPtAa;%uYnwAC==+J0Nz2FDEbDJb2jm_egOTAR9*K4d&i-tI#c7XzgZTRY?fGuZ)HnK?2*qzK;{xxA$cmG58Xy^FVoBcP(yXtzk'
    '`L1y*?G<Tf9CiurzA`2CEm{{=?TeayQMWG|rFC2;_`j5<+HLo01!tma^BxV5PoRq8w=3h?)}(b_+p5IVt4e=5!fAk=B>%%+qn$+L'
    'm#FXk25q_D?>9^LAD8ZLmF_>`%U0?B5A;K`uJ@exEuNNk-X@(V{ZG};pwsfZX8o>PzZ;)}^fQq7s-o>-P5-Ft9}P|Z^*aOrS*_d9'
    'rRfokC;q_C>ZHNSEZw8&%T3y*v)fn}yjVv^Tg|WSr*8TIIr!i`FQ3GrvU@{Rr;W~T*X)?iUKexkyyy9B$~*6E*5||1lDA1Xx7niq'
    '{XlSH*XXnJlB6(homMKX)6FN9TM-;`R2LTL@HELz=Yud!=RNt0*75gL{_q$KeFMs_jpq)sKQst#4D4PX6!+}hd+0S0C3osQd&b$T'
    '5IVJu>j&@wQEo!%XXo%jCqQkoYXc3i<Jc$a-X&g;rTai{(fi^~flhqt^cT;kdk?%PBx^l=O58g=fL0{ePR(ftA8pErGOd{>YqVJ}'
    'XtTAn&DL6NHVWGOVQHH`ENe48-*I(~fJpFzm(>c9*vlIJtzJK_HLHZwy(&=zdV?8*Yq?R=H_mWpc@s|>TXp%Oo^7JBs)-wJ;>I0C'
    'Guy=DHJeCBv6XG&$?_)ZJduVAi&_>IwN<cin$V-DWnocU1&caQq~XG%mW4%a6)c=4(ot+>o5;XojnnjWwh&FYulXyHgw(xt6svJG'
    '=+Y^s59Cm@EZ0SHW2#rEgo6Q}enxf3lG;cK8!Q*yRmY2=TsHF~(LZkVkKgo<t2G9B*|f&_dnzZ<|C(P4MlFAD)ZgFK->ckgR!zq^'
    '{OG7-9%g!X-pT8%iF04TK@Y;59doMWIU3{KVan$igq7wRLD>J2|Ncwlc0Wzcrz5-ZLl8~^xFz2D>wEE}*-zg-DEzY(0Dk+nA_70|'
    'UN^h8fIb++>FNByi>uC6&AF;OSB)Yh9StN(GQ(@OCaubKv%--yT9F6oF2iGfy5B6;hKd*$Go#19YUWqn{Aw7z08PrM{J4fg>$=6k'
    'y*#+S4#-G-!E@SBNg#vlINqtz=fn1=LRj1Uj2NgfF;L^pCmcQS=#e;4k<hGSMaLZ-CpuSA?d(4Ls992q*rq?f-wY}h`McRL(S_2$'
    '?OMTP&TG1CMV$q8*3((eC9A1pUW-@LTE~<M@luvRf>PZZuZOVcvF6od&Ckc0pQj-dth+;+v^n-KB#~DkS>=~H$NM%hU2LB<$>Hf{'
    '7gC^m=iScXTiS8%O?cY6cV7Pd?qDNhH%^tm@80bY2R@ns=hiX`i#?d*(nOuU!Ys^hDSf$Rzxg_YFek%t4q>-TeqND|ZS43fy`|m)'
    'OS-e21=i^G&OQV(4DCj+WDuzrfir}xOTpc_gi{4><^hPn=Jo3Lv@jh9)Qzn=Cxm-j9*4ie8!PX(DcDd}v`x{bG+5vBTkO`aqSx3%'
    'P{>mVkLz3Yd$5^tbKT?$Qnf>i@_HD&9{5ZZNQ|gnH(t?y0Yo>FCH^28!@N|3fKl7Z+!Yr_d7Yj&>P26H8<hj#<)^LY^%gO<{${Z4'
    '?cl0GoU$*qEz;xwbh<uQ4-rf&kKI!t1zE5rE{(UElda}CK_k^oJ@f|uTJ=y)d^QWD{IA$ec(6iS<Qm}Nm}C1$&A?|qQ8qXIiEIwc'
    'Sy3S8$k1S5EfYY&|G3H{&t+oie~;;ZaIK=g?ziZFKj0mH5pUM;Y7Ot!@N(_3rHs1IE?G5vSi_e!d|Jn+b?yY8*70c_pVrY)ot?Pw'
    'MIB$%@kIk)G-|GztHDFW#|?bkcnpPfgT5xXP~$b%w?OUKfu;T$Sn98VrT!W~{bhbPt>2HW-&@x2C!ej@7Im*>J^#V((;0w0f|@;q'
    'nmvY^J&4-lg5GNOaB5HN3wt0ndn9#xBy|TC_DJfDg8u6Ei0U?6>h?(L_DJgXNb2@T8umyUwYw@GNqE7o(&i_%t=1zvODW5L10Cun'
    'Nrn^&g$V77-Zvx`&HqP<MM^9Fmp1(Wh&Eu~hnfKNmTZ6HNW%dE{Io32%bf7DIMH|8P4KP~Z2Cyl3?FHgMM7l)IFXvC!(>BeujD{>'
    'v12C@`$uyT5Xo%GV#-j*AHlloq!?Efow$GPzKUph&WFZ1xK$d>4GyxFbm=C*K24s9$dkoH!mhrSI(2_rgFgK)N&452q-~2NZCfO1'
    '+agKZ7D?K+NYb`NlC~|9v|T`wwk?vhZIPsHn<Q=9Bx&0wN!vC_+O|p3b`eS1wn@^qO_H{4lC*7;q-~odZQCSi+a^idMI>q4CP~{i'
    'N!qqa(zZ>Kwr!HMZIh&Jn<Q<oOOmRF7%Ab<_^vQbLd+TDM2RcfYMSsMEf6A2G^?|as6*Th?;;CsHUH(JAO^Nv608?Uf=`GA+q_i{'
    'd{TG$FAYD34?{iyk2CgedQFFz-hu8tsTW9x>Q_{<taM@nbx0t1&y_@Qed@-;_@7DL@K1qAW=m{aN@Bg=z{d^vE!=Nbh5r37SN1=e'
    'E6em3S*sRVs}@<S7UkVc)K~nomj0}#KLyBI;xpET(R3R7*T{V-89%CUvby%3O7c>W+8XW*^c~yHT8w>h)E~}bjwe?I<73nU1e-<#'
    'X%@86&!4SKcJVa!8ZG@<uc{JQx~h{d;iOZakHa}zI=Jj}KPp>gEKoh2MvuI*tLPSlK`GJA9=O*c9;ShN;#fKyL*Z)A32tVU!xsna'
    'Dfwo1cb7BMmD#HtSs>afPl%bdgiVLNWE%HW>V9~y1#U(JFCK`u9cD_btzWC0z-<fu4C34sPyB(H7BSF^)HknA*WLIxKlAc6NPj#C'
    'Twjsc!+MtK`-kGCc!A@?kCAMICSS(Sr(V@IVO}WYNQS3lR^sd80*m=<G51ahLTwK<F*f<%c{n*^6r$tHRQRBCW^jf0X*@-$T6_6B'
    'fR6I%`?LN_nP@G0oipI@_48M?E!Exk$~Ta>_y9?Zqw5*<i<D#5D&6j#hq)BG)@4_K{-O4}2qru6%hzV79ITFrW=lzsFLc$9I8TnG'
    'k5m<SSa`zSi}TAMUkCvuJYaL{Y;dM|t2z_6B3V4yx%Fyd#b4_xMaFytzy1MrSMi=BsGn=kOv^AUsW2&yaghn!?m#sgjFwv#3QB}_'
    'zQT-xK4d&y1L4FsJ5-0Aqi_W$Bt`hqjjpU@<^MFry-Yo(a+z<W>!#AMnYyVe9D+cGxk!J}H9@(<(6XzQB8N3+0o-}?oQXnoasnS~'
    'CU~ZBsct#A4OqDe7eUTV(x8$ts5A_^?u#mVXzHez6h^85jGGZbrMKrgFbqtU!9j&Fp)5U$22NDjD3|2wM93&RvoXJ!MYMgG+6=F|'
    '^UkVu&bZ!5f*Ru0vp#vfFfrVtq{MjuflsrJF><>|P~^ytDu+YFtinky0=>i06uW?>f$tCwh_#+oMsW}RU+I&yJ&Pc!7Vy<W`p9H3'
    'pnK+hwMzfQV)zje-T83P@gkzF`}nho1av<I{}c)t>bi(!I6p}GT9w%x(S|pw)(*nT7>~o@$PO&`m9?IiCAyCK3*0rNKy1fc3g&8<'
    'L{dw<QEsX@^hi=;b&w}2JSVi5+p6a2R<&HKR=xgBABi9NXw4cTb&SbRC1i>uG<bEpX>v~Yd~v6wNxgdTz&pW;{l^Rrp`n^DLOuXK'
    'ycfZRvD(330by0ckQdrM;s9+tW{~qUpn<>d-Iw<{<<=j6$YKzi<43%>Un~a;@p<P!0bH{9AWQ6DSM@H?!2`&*7!qP|<*fDG&*9tO'
    'JkbYG)%Z}=%&)rn)ktFtYFl0LS3et?00R{?KqFvvSfm06(rB2v6%liG51#BgB&MppscOPkd{ujJnpf@Bb$boR1uET5qp1;S`TA_A'
    '#9-UGh5h{zz9BIhM8J?82EgX?FeqFV6UZU&tRDs;OOXopXI0x~c~);YArV~^MkkrI$KZl574ovHj2&F=IPj?j7XDA@zePaXs&cdJ'
    'HMsX~F{pY=+Nw94wt7q2qSw;4Tp61n$E>0X6u>lS5_J_Ta9#Jq)zJdE_Gr;ow%5C@J&hJLuyk%f3<n+7^Fyq!@Ize3>#&~pI$Y&#'
    'co*+N8O6mS9CXLwwc6{KGN?-e;R)S6@h<_L<m_bBy`D-yq%R<0u#`M{u6F00zNtGq@N%t)6&`A`7-xsNxxK~^JL|q$zx%+Z{rP{_'
    'xYl~LnVW*~kP`Q}GP^mZ`PqDxmyENmNtP_Be;_H;*z6&4v<e4ShEWzABq>{PtAiD@Kcg;+N&6W_DHEL*FY#;D6#l*%c(+fo1vQt('
    'D#k8hM}n7lt=uUifn*T_Cthz$;$JzGUq_^?kba#ZBz2a`zQ}nL!v4i#7)O82lZyrpnW+e%T455-n~*j5^ifp~{z-rT$<?MJ5=8z9'
    '82<D7MDjh->gp;Y|KzlL{{(ZGmOtZUbhqw_CPp?IH1iO3vMh230%Z)Mq;tNhpg#2^IwdJw-O8X!h+OYF7+<mkZdwf>v2X!0g|jf1'
    'PGI4Lz}hI434g^v1<lk<FFOT(xw-b019_1Pu89V|tXyk@!)4_r{rOw^^GXjV{t61pYf}6F|C+41B${PJS_5klRzUdHq5wxI+WmW)'
    'snk3F^y1|7DZTjWS-Y*W)IPKm$bPo+ntGROzs+JLa$kIEeO3wUkS{4M1D__J^|!QtVOsHuhS>w<*98f^On#8S?q-H=nx8v`3G0xo'
    't}F%Rr2L!YQSp2lj?Ni4JM9<ca6EX8^!82@oJCBRwH!`t!lRS7=v1_-?LqH!(sA47N_ENsxor~om&ajz?j#nx91gBK1qmM|0Ukuu'
    'Gy$$P5KI$gi5?<R;igGD!#IeFNQC~X77~yU@C*~56M{1=s8a&wr&e_ZKurZeqS1>P3EyeWm24&#l{(g0FbL%-US!%Aq|mDb`@DWG'
    'KK8nvcH-15(TQOqrz0-wEj=ykNagz&h_+n=OVXOEOdxRBcACAMpM0u+?qn;XaET}Y9}3BNq-tT><yLZiWb#gDdWlx{BE>jwB*!S{'
    'g8m*=U)i>o++{i!IHEhb0xxN<Zq7d>aSJq^3x*t4+&`+EUtR=KXcW*Kcywoi(tnl9iO&9t>KA$<DtrIQX}kTZ-WE0NV#2RPQbNQo'
    '+M`;ST+e}#rU|<RQ`wijmA_A<NvA=7FkK*5rLYY`J0xSt^(X9-B9aDA{?95WI%L)h6UqSy2z%=HPGGpSRK}wl$YFci+p0ECe-X#g'
    'WUl>o-ZG#*fNrpmUiYrOI_K}^e*)=F3Ust>Fgu9rsJNk<DoKq@om^guyixnA#07EHa08dhE+vykT<L%v*vLvM;FlQF2$P=FW%8iY'
    '945T!wG23>lPIn81dSd_vj#FL4o3E=!!aw=xOlVjggQveJW8h1)TXX{5sddi+XHx-Th*4F5;oDQ)|)|klwDqYva;as(J~Sgkgb>g'
    'L1ht9cz^gv4;%F-TU=^D3Y{Q-zKsT4l*GCr^+_6AEo!Un)mlyHL+Qs>qty%ojVQ8mb|D>xr>dFKPEiM^1(Yurizpsu>r_z}3|s~w'
    ';9W|6`7N*nOW|<QN)#Mgxl%5wc8mYrD_ulH6?X|R!B*{uI#27fq!nMvu4R*`Cz5DUu_Cz;cK7vXK8Oaw6f&vHX$6z{r#%ke#<J-H'
    '0vqrg+l7f-An1N+-C%ezVo%SbNv_$`e5?7q;isW;eKBd-<TO=QI>|SUip+ZtYg><GjRKpIns6TokT#?ay-cDj*$6i6AfE15B4}0y'
    '|IdsOHmhZ~XBfwR!gm{W>s~U+-y0$~{HI1rb!9Mz(}P{4jwhmyD(gshJm~`ieMP<*cXgVrz)CbeQVf>UoJuiX(pl=A=+8b)^wXXL'
    '3zJCOg$61nVc|5(L`=@9w$pO`x=xNY<EnS{wkb-g3)d=17Uhenk&+x*V%PFDp|;GbELemov({gXKcSii7c$u7k~6i>=}=DfLlh>!'
    'qfZU<l!j>reS$poC;g}}4>cEIZWmYV_El--@-y+HB<dPi#f}CylW>eWAV_SSs94p1><4aDH?V+qnPuhOsShLmxFpHBL4g*LWL`%u'
    '$0RL#WGX87Dl2wL*;8q>CAJFk4;(QalQ^51OK=kbVkn)0ADY!(4aFJ2NQac9Zq_XH1%FZ7lDC_#kTJM8u^(riMeZ}?eFpnrZT%8W'
    'm0%qf*8`vvg5}V58x;ePPNEHSoxOZqR)EX*fkbd>D9(+}swnow)i%VAs&WRSp&9|btQcCZX>L`{>xIN)aBGLpA06)Tf|(5X74N-W'
    'Di}4`;S0>MT{AXjEfOU5o6AB&umUFmE9b(LK%#Xj>v7O?;bOE524X|d!2-^Qrwh0rvuB9ydEjxD?x2zvJ7L5`W;+r=o-Hkj%(AK0'
    'm9e2TFZv(7d*B_FEbG(@=SnmH>Gk2ki#r8ISuIVks&-4gnen?je4hT?vvQ8O#myAEa1xG^h**aGW7lzHJ{+*3%EXbg*#Z!tMa7eN'
    'p^D#H`AkL&D6PfICXE&^5fFmhyWsV<`IEB#gf3_=Rfv+ap;gLs*hWFyu1cU|j>(u;wPi2^CHe-if|)lGXG>kPSDO<0eJ*`2l}FfA'
    '3_Et<_hxDsZ0JSJh=-a>V{33A%o;cRz^gsRb+hA)LF)^tgJqd=<P~JjI(+^Kl`xv0%?MTQ1sYZf<|q7fgMWU*KUeCtFROuM!AED6'
    ';JJ>t&igG>1lWk;9PWLJ^r&*#EaYY;;h#3cG8%CL0=+o!tYDvey_I!Q3`M|%yr2#)gkfRW`2bXD`D0aiGP~5+>)Y(}JM8Za1~SP$'
    'J=pYkmM@MlCEkBE1K^_y&Ry8p`15Us#fUJn<^Z>Fda@6sg>=GnH=S!CtiA%F*GB2wPJi)Eia6=VTF3f{p3wP{Gr(sC*cU5&AExsc'
    '?>gS*X5S8T$As}r_Uz@S{%5uojt2cx|2A~LZNsg#+kzlBHSD!<cpiy*wCPvj*5&*YeBHdQ)yU5LcZY_<P)rL5a1hyKgRs85!R2q}'
    '@;6*wnaeA;ns}aKd}H9P1F3X4UpDf;Rt(W1XV{mzr7ySbfkN~bwjN7Mqw1_&FFrT&3#PU4SDM!6>0cr7vmp?QwUE(!bmai&Avf#{'
    'E`<`4&mthW6jD>HPcFUs%yTmorGOHBN!98^ezbwoW*QB@RPlCAeWFYmr?W6Inm{n%al}H>z}T_;gXeAU6XeOypCAEFp{Mv|%+8w!'
    'd-{<#lG}kWgW6Nk2|cKA|6(W;Mv{fBPq^_AJdwvHh|OAntH2~Y-mt`AZ9-)utI{b&LwG$|g_pRFkN734&n#Y<9ZF5ygMM2x#OO<z'
    'XAMIPcjQQ`vWnSa(Jc)V#a^cz@LibBLd&G^o$%60&emW&d-eqqZr(WpKmB~tg-W30q^=`@=PP*2e1*ZxySh?e_WflQ{ch!1ZC0H3'
    'K7?Ih7jvItKc8j1rg0P-Mpj6IVPajNCfp>-Y{4PNeNNkW+v|bA@Qa)LSLxo4D(5We991r*hS=a&!Y_CL9z2(g^blL@587CANsKYc'
    'agrhx`ZGK`q*rOE>P?claP>JY33q_%hQ$<`2s8~Lu<GnlN89*f)7C^<&kxZUy|l^Q+Yj*JAwCoiM!w2ajof|x=KVgsXKOiCONuAd'
    ')#ciz?w+b>r@ixt{HSeCqoC7ZQ$Z=PGGTEE@_-X>J(1N?o(MC|_2H55OR4H-&^zr{HI>aYwJIMNR7U4CU6Tup8w#>6_&7<n1?3*{'
    '371Q`t+l|v>pIe7qBiri03cjGhlvQdC@(1P-ZQo_u}7||(47-_S%bx4wIoUpp}LvPl^0n%<;Q*Yr-hM*h&GWaWcR@;DUM3)kWij8'
    '%7U7L%8l5q8dVXcK=Bc~KO`Oe!rb7OI+Q{X^a?U?raf{IF6v1JF7g07Pq1lbF~)Eu@c3+Tax#Jf;rME*1~)VIC$UMv2bG;%rLqmo'
    '6(d+2D;bgV`28^!Lh;#8_>9eprK3J}bfZQhK)2f;{pRzociBa+=1N>~G~-g`25YdIC|FzNBCZ{I;i*>#TY}%3Gf{&=%~Kj$Dnm$v'
    '%o8kk+o`7Zd-^)im8J)PWQY`BQU-xFMw2O@VWxx3NFO}#beFZS$9M}O=gNEc9NxR>bWkywOC;0R7&E`Hxvg#*^oLThtW3U^bM!_='
    'f~hZ)U(7HTGKz(U{i0Z+!b!l1K^HPO%ahoBkujFu1+5?97`fo92OeRGB{OE^H57Z}FCTe`8Dv|ix>Lqe;9~mW<)_8xyYS^$&U2a|'
    ';F*(9xGxDjRqZM_A|i*NrWYmN`<Qz?GBN^xMT~n%k!^?3<^29=j6GPD2Mn6q<K|3j1s{Jg-0xplHuIJw$*pzx$5U)&`SnyNsWoIZ'
    '+FL#S__dPB2J3u&v)<=7OFsW?z0ZGJ^7++zpI>D^|MKsS?H2Ze3SDG$MFn_#i^S<m7h4LA10#R}Ksh`n!7LtqR!@ap%%ZCM?-%}#'
    'e&iY77f0`Q{tk@X(na0n6mQG1h+|1*D5I0KDMAScC-tTSLK`A_W8*%QDBFb)^3`X5uyJRFlBG10Ow$={(h9|<e9C&@k!YgKJKIJ~'
    'y^AI$mF`DIykC9XUZ}(6qN=w21s@{K6m85A(fsV_dcBd{TATbdbL6L(jPiEk%~wG7M(P`e*N<6}sXooDfN92S+e(EmUorGl9kp=P'
    'Q43ccwQ$u@ASWHH30?6FI?iUetKtW{c*Ke*C_{LX43khoag^>9GT^@#KBv`$>FM`+Un=gChC5B|VDiF9m8DlDyb-$^Y&b~CgyOS<'
    '`RUDq^-U39;XR{0y^a;WFkBNVv@xamKFbF~SOU?olupuyi`W;1{J~P3zF3aa>!Y(}zxbv!V;#U1fg{~47w%@ca5u|^yIC&W&2nKj'
    '%QxR9U-A;FtfA$|P?NUq#K-q*fn~K>kw^Mpk_);f+2SE}u|Ux&o+)JamXYFX&8&o9*Fun-xX`N%q)v9oVc3=tT<J9Bdv^JsnT8RN'
    'RBY@l7|#zu#{`~gle=0H`PMNQKizwVfgY9EE-cn4)_00I7e-uIZx8nZhc2czCAb%jlL;iFhdM&vhc8fIyxLgt2(Ktp5{`AA5b>2m'
    '9GnpIvY^bTR#kZ5sz>Un68f(yNa}%O3|1cItZQa_8k~1(pR=alo}2{8Sb#qdNJd8i`ks3(^vuDqQj{bbt@H|#bs%p|<e*ZN*sywY'
    'USjNkl4jdIn>GqLV!ZgHA%~!XC9ltg?D??itTtT7qx=iM%42S@<ei<V%Y18!P(rDq$D!C>hZfr}4<`sd<4>O99s0z|lW?h=sPH^k'
    '+K7;H>F+M4Z)$K&*;j@ANzsSjQGY393Pht1N}gQbWUhb9Twkf{FK(j=lt4EqYVlD1Y|@{aT-D^NE?0HAYRFZ?@SD`aZ6aHe*p}!u'
    'Kd9gJgXUd7Xaz+m5rdS)$wfRr@u?xA^T?F(h}bGx5m4}`{P-mEr4(m5g*)W-t%ZUf997`=$l@FGYW=3`3n8yNo|5&4-f6u^#a;L='
    '^?|XN24#5DOZ!@~RLS03Rpe}X8QdmDoGTTW)?%FJl~0n+#XFs|)!d9!a=O(0)B30L&p9buVJL(k&T4Wx=P0M7X5F~4zH*`9#dW*N'
    'o4Jn+GB@mAehySdWhd&R<f(gD;q<;O`cJPuv4wJVQMfq8Tif<Y<RMt{B^QsX!2646u(EKoeRk9P73>2&{ON*mg)=y?O#${5x;Q%v'
    'TqFX6rPujO(HW)}#$uuxdc1dIqZ+A4!v7UUN($aI6eCr)=Zw?+u}OUZ^XBU%KE&FpDJIPFGmrPC=W9pXl1_<$1W1w*g;U#$f+P4w'
    'mQHr@?Si-Y8f)jjn{BetqzGeVP|_Bn)Eb%EiCjt8vTRyhM{9*Z%-jdF9gULI8w1DjBy0A_up6b@*^9J5a$3DPPs7hJJUh#o`fpZ2'
    'nxo>KJ*$icUG<nZxzsshD{CsrFWd0kV@c;m(3t)n*<iJY8R5#?4+=@N=kX`btHh4NLB=CZc|6ai3CFG_6*PE6yb&j6>RTyylqal&'
    'nJBATt4u66iyPdoYO~BE2fNRn<le8kwB4CFz_PC!+z~3?3u$yXVTS};gD7G?5~s<1OX8nTPb0WO1QNWc4T_e&F&Rq6DBAQZEQ;)K'
    'QD{^o;yt0sU4?138V`Q@`~uPp{aL?O6+Y9<UwPNt+sOfOnOy<wvOrty{%SGD_61;S(*|1RySv<Kc3)?k@E_a+pIV>Q<paeG|8B{M'
    'M=xqpl8I2nvF^%(%x)wl6DTLcG>Ie;6F-tTDU{`o;xgYc$$R{SL>M4tKbj;9<p^mG)8Fsn+iJEP>&vZO1R&|UgE_wbI$Yn$k2(#4'
    ';#PWhYlZ6z7D_h#w4sb+S8jg#d*SJ~X)2(k!qfBBO-J$ceDzyLSbMp;O22N(y`FsiSYPqiTl$K>exk3;*DO{+Pt@0URw#*h=4)F6'
    'ANsen#5Ee%ZjKokV1nhIHu_Wy+RwK4oA4Xrk(O2E9KC4S+oEN*1qTUh5~_3Ml{L-2=#Gd?wve@MLi<E{5Yxu^K1tP`Qsrua#AhVx'
    'XEsK`)`shYXl|K&HOpr2>12e{={c8{PBX{t)vYd!8?xCNOcSnqh~afD4}HG1T4b=i7+X0PMklGj(tRzf(@NbL*WBSA#(o#~U1w)|'
    'nDSk}c6PJ}2-z*9Tluyrt!o@v3Eh+;D6id=C@8<(lrAXG-IO$Be0Nj&VA^_Y-EgRcY3qq~!)hy~7W5$F>-;Z4gKtKRzBxHsN|0`u'
    '>}S_8_sbFzRbX^XQAe2^`&Hv$+;@$RDFJ8-0cg4wp8hh!<2q(X%k2_k0-Fr=bg0L>IE>K9?(#Z~dy#>kqNpI^MGo+HAycm7^<5~{'
    ';xAFE#b2gWixre=<*c$;_%FbSTNcQobc-ydTVyHSB1`EO%P8F<OX(I_O1H>Ty2Ua|x41K<`$|muA4}|3&37ri)57o`4UQdbwR5`2'
    'kebzEze~<jYzwwGZAfB}N#ZeoR9a?;BH402jUx^Vh@(VhFgA@<y!|Oq7c$;}T~kALT7)i1g+43ol1kF+<0%PH%A?ViCskJoyFH=5'
    '*@E-k%$2{432r@l_kEI=VcRR@Ur6CYJB=#GjuMv$ak#*lU-V12i?amt?xbH^Da+<O_X`DG+i<p~M4L84FwIt@FZ7;A4(hC$0g2AS'
    '1o67!4Z*HV4qwcG6lI`t*+!~hm3IXT*0D4hb=?{*d#F;F2t&@CxPP0DVu^yHZDa-nwlq~IK~8yhuyb_8%0A_qPBOv%G5y;L@`}FK'
    '2k?QH51xsSFDIbL#sy&4dSW<y(pkw_%oJxa17{&E^92#Y*9Up#;`M=W7xXv7UYI*jo;TU*BKOujVO}3BJz`=ck@shO$gt{N!ADz1'
    'p<E1>SSVvhgBUNfPku8^#b4nK{wim%lrLDVjNqkOe7?m-CQ*O?BCV;0qKPN__eGj-i!L~1zcaM(e=JjQyGkatyA>Gv8E;ov`td@>'
    ')UU1YEn|IG45edXS3G6Gz26G%X={A(X~ysM3i{|3^wBHmqi6R~Wbt3-NR~0fRK@thN`xp-cJ&*Ccs}Gmmcf17B(qJDZ)R8j>o#yK'
    'r)~Cb{_D)(7QmVcV0SlzNB{k1@TkBH9<4NkM_+9Q7YbDTY4e-d!3}}WSi+5c44Jidj$iz=|3;GYXx@qQ_<#>X;ux+W5?yg|wHb%z'
    '5oZ-L=Wdm5?1vMScRZUSnGaa;9{5737U4wKD$@s-(jh)jbh@1DZ3%a+(t{oxAL`Nrx29_$`p6^gPKN%`yNBM9N<utT<zeUgFI)d4'
    'ZPJDOh$e%QW{d@C+3Ji%KAL3$PZ<rlGRvknRQdSaS2!&@5`)jH$l!Aklq9qIcsQY3^cg-8`=fR!q)(oD$hZIdiVPpfqF(lp<-f=f'
    '?69R%&?9P8`{kkta9>-5OI#`l@04je`GYxO-)O(@{eqG~^g+Ny>r(fk<>-r8*G&SCn|c+7W|c4uPp^VC7$DBp2nL6T2`Hv5^!eno'
    'jTcsc<#&WL!HN?;?cnm84fS%pp{5Qgd&(ff*+RM~M|C4McHek|swjVj8LLw0?(G~4_8z_mUg6if_b+!rjaLe({I#-*7x-O_qcJBL'
    'q~@PxT`;;?xz*fIuCyd9u5Ws^P!Ai+X`FV6b4}oNt|eZ*`!$_G)zQsGx;uB~y)M9*9RSJABYCj#6}IK%d^qc(0tcFqKnPR#NLD?O'
    '+nl#kt4P?QcCh<GW#=vFLPn97%%Z+ZHcyoNlt}D&NCNNxHfEBjQItw0H?Yz^+(dH->1p0eaquCuE3*L34J(fclavngR(+`CZ!lWx'
    '*Dk}+0$Ff5k#SV6dcAlW4zOWVLb#uI4qqPyoLBfV?9tS0pIz%%Ns}!*<%d*;V&q;Jh(bVRanMPF)7qG9?1R;C9Lbw*S@LOYBG1us'
    '$u!VgAn7Z{7fphV|KLKzk@xfd8`NA={*-xNN-iT&4cv?Xw6^%J*9A|CQ(CThv#{b!R0p^xSRY94I=Kb35BT)Z?-1H$qRj>M)pNBE'
    'f|5?Rg`14wxCO-4_ZZ2ctkFXxBnNLtCwFv$8ww|SZIw6K_NCYLmZ4|~N6;VLKw;zg%sph%7&8uWD&rVf$a*%yg|SM7DCGeX9YQym'
    '4<`#+@_sx`OrF4m(ZbFlDEgt0eQ1o*k5lVvJhVy<4&rc#vd{`!K^EI49j(cuhsx6!f1ve{G+*qpc}@+bvhqzOxi9ZT+Bh+%N|ntj'
    'IZ#wtF`c>2&z99nL;O}MwrH=^RQ0xGHX6>$Z$Fes*4NnpOVUll&NJEZz@QAG{ty`?Aa>B|aP28sBAH>L0K%ol_6a6@VEp01Pd!xg'
    '0uusXM3hXM5-}2{jmJS8ACVxmK8F7fl!B7aI8%8{rGGTl*-9z78BWTq^HW{`x`Y{@9Fiy)<uQ=Jcf`|BWwg{(v}26Gl?yboiwsKz'
    'kGxA_*d2`?(_NY<P47D#Xq;)5=llSMz3)`SgGHp@bzkf`Z|m0^fevtoOCN$jkgV5vSP$}zy?+mdVqqfy14fW*r_ovI*T^UHdTx#C'
    ')grG(9w4&G)7yHo?pGg=bdFxU*0HjFYSF>UE9@rxWC~BzcsQV?6I{^+`^HEY&4|W(NhP92VrL!QqBDl)hf6FTfT_7C8r^Lp#~0a*'
    '!z#aUCIRtx_C)XG{F#pLV;=M$k2+Ru!=FFy93BX$IIpUt@MjS#dv?0w7N4CF=ghF<U34Z*`_R;s)j2WgL*Hc4MuFIPI71!ilFGB4'
    '1cm2m$9wVf@!rnS(f2%x>10UkLKj2m4hPqKBznWaa5@fF+ZjqcJ)<#4No1b?wr8NidZ{G`t<okw#%b%d&82%Ro^ll3gyo((zA-l*'
    'SUXC1IVNV?%DADjczmPGKh`-d$LKZ_pdW=^;G{1bIT8XP$JrC@VtG<|EHP;gIfE9q7*qp%<O`P=4RmFPU?Z1j=Hrp$A)FGBgFFE_'
    '=oGwofGtLEKZ303dQzFB)@GZ-;!@PcTO>Te<Q`Fb9lH%>2(og%D+LLYBJ9JIO*W2WymxkY4)^}?eCM!jRZ>+)|M!VbvrLB;vUkBG'
    'xtMw8+!Kh~@t}~IOllM>g{3NLK^y5z@hMimFU4s1s`TsS5IJaZ3PxNSg2|yeE}cpaK|Z%dG8e5`f~*IMpm>{`$=pqi0G>&JF@f(j'
    'm#IkRpG&ktP7aC8ds1bBUv$^}6%1M}oq7K62i={+!=1f7y2&Y9ScBiFn^ZHXXTWx1UCw|Pj(IRtmP}F_UXYN&MNEk2Zo0m;!e|O6'
    'hguisc@Px57s-2SQpK4LbDweTiK4I4Z}<z5;qjfHQ~52;;g^MqM#GwuJ(UZOWeRa5ZBubZ*n0nhbu{d9&czTeMB_tr<>2t)c_%Fm'
    'YXU%0{aY#R436lCmQBQp&uPCqPAYUH_cKZ3p&?{_o0OYT1x4(n+OD*HVWK+aXiWpMWyifbIDFrw4e;*3Z`KRUMI6y^q4ZZ({*7to'
    'UL;C9XZ=<M<zTw=5Je=kLX`=1WgK32@h~aqh0D~690q+IYX+gxF+=8}Jf9BUrn>(*Dn)le&*HgHWb;04tZmjB+&g%+b9Af(At{ca'
    '7Hy~`Y`8gak>=4z?C!G2-%Fardqt6`GcJj}ot6m}CtEIsD?djwD?>-F%Hxb=4K6jQYjv_r3O!Z5rNb&L?OOKh1lmnn0HfmaNu*>|'
    'VYNtCsDzB;cRL5KGZlIWWl(s@7cR0UEG49<HWvvg*DKBYB@eSI*D6o$ck<e5UE7(15UTW!6WyEgW88a}v(lNtFjFtt*mS`OzTdKk'
    'UNBxEgj>xrod7XP++3SOtmGZMR5EiY9M-{+q*N=VyVwZ7_Q{t!IKe<RxxMWa#|8I#UYf1Kt_9^3fTFQxt3IjRI*;Ty<5I^#p$1h$'
    '%O=0>8)>&LiI%|~RZz7mC6|(bC!6S}C;`y2bMe5Oi5VKr^|knTs}-FAMWi`P^>c*@E6mNJ7H(Bl7Q8pP=hGqMmIy7*A7td);#yNI'
    'I7@zlNQ*o+j&(c<<0qN|#a}`~ys8V`<9CTnO#7TW(Msv};^&t;))OJk^~ae3ut(1DH=)*}Fh(JDRRWxKJS1Dtrjv@;&el&OC1>;Q'
    'd^!)AdYHSxtnz#nRdqfe4ae*&pc~r6IrYH%v$3~lf-o2Jsd!*XYxMr72cH=yt}_^^a0om3ny^IpY!{;}USHBv`M}XlG<6?nm3uWE'
    'F4v{D=9xJ=nDS``{#qG=0-Q-8z`?4PIRr4OW+q<vW=)+#0=p6k4lp3Gq0d0Bpf8q8xNzOJ)bjMNka6Rx>Lji=`-FCN)TcGMPWKc+'
    'i*{97AYK30Ta3=T@Dt-#QK*Vv{e&{RnIo6eT%duH=QLZic{*!TB10jrg+O5C4_<;kTn9GZ_}4jZj8O|?mp8g|8@EcrJPcgUdu{w-'
    'IHEdf7rc2Ej(++XNkT_hx#Fa1ClDu>qgBx-gOOQee<iw}%mP*|sk1{-Bvsf{>1A{>veW3+#OW5he(<({U(cE&vy+0Ut;$R)!p?0R'
    'qcxC7b2h0xO_?%Of^x8qU(j0))g@-s;6M^ytiZAiA}mLPe+L+(q>Z_|JVfAxcIt!cRYph+kl+?DWDKY2Zc9Pq(;l9kI&orBDd_OM'
    'xJBBj{M32GMI?jNKBW?A<xe}Q-Y4}GH;T_=9KT#rlz8syXH~P5`ETe;KHJ{@`Xkw`LpFHuS5V5BFVgpYIOzs>F}-ICA&+zP!!AL{'
    'Js~9RgXeRZL#0(qxQe~0gm{CU?)gttF+D9cx4oi@bzFJPE?c)@t6Vt!SV@F0t8!go(Yhst*-9#%`3*Elwu)8Q4tsYHI(6qH112-*'
    '{?iR|vz#l;N%po~m#f>)K?83<I#boDeV*UAnB<mpB({Z%Izo1moNN=Ps3WsIQY@^^X3j{L5)xr5&mKDDT8aDB3?J#ZVq7TiCa!6w'
    'JunV}j1hRHd&K31amtY7bf_M<={2iAZ0W3ZjheXj1>Sand6^vuxk0~r-Dt`GEok8DT&&i8IPg#^rW-!J^uj$-3#Se+wM@Jfvio?c'
    ')69+KG*gg&zQB8CvdVqN#-9m$OHA02CPksF`a;>MeMeuRyqv_CC@PNNRz&b8?2<Fd_~aa9rJDRfFu9v^&id{wH>><viky0ivFw}q'
    'yc8kZUl+RE_JnlcI0_078R|bK!;Hx{tiumP+ioE-Q9`glg34t&*f^b>@q7R=FFw~tN-YIj#*96uF4X3JfJ7X#ODMkREp>GzSFlaw'
    'H-OQyu(P9&x1Vw3;g7V5pFD2V7C6zbx0?0pLODQYUgAgm62D%&$7XS^`I^fr<OVdw7f=vA<T(J9XtKXZBVZ~OHi4A!I~O9RJVa-!'
    '+O2(UyFLUOYJ7?GGeaVeZ7D!7yW(T^AsQbk<tUb(5qQL#VZ{(Hlell!WnZ$>&>D9t`@n8c70!I<UCvoOq7H!@8N@@jCE^@dTCB*T'
    'K4-#lo@pid*$Osw{Z!koKQQs6M1}~TW7?2$BTIxvgJj&jZg%HBg9dBnV@1wXAi_eHQA3*dTfl2NvI$Qq(@oG>H4kehr;?!%f7M)0'
    '>-ov#*6|Q1d<wfqj7T1IepoZN8n{*KY}`%=Eid0ooM&h0DL$>LN*BO24GJ0@ya;FUc<vu^r?hpmY{uQYs;%2qE!WkTrIgwJUF4f6'
    'v=ZE{_9K(h4fBdPSd~oa?~DQ8;WMOujyB;4ZC<4mAg26P_s#yB<BjDi0FjWC8lX^qpBoPJx`+c6<?6)N?60Ga@LZh(WVx3Pz-@hl'
    'f9*(9g}PBo_CjkoRZb@w@@d+it26e7fjLr1R?3rx6~G0yDhA@aox`_&JriXjFu_rvnY;WUY8euJ-#{^C2h-!Upu=jW!)v3qD>D9U'
    '<rE`k3!nE~LR{o*OmyY4S46Kfuce)&{^=rR`(hr0$i;H@pd?Z&w5M1PzNylQ(y8?Nxk|m1oDEOrWt3nwSy)wCANp<gmL~Yg6vwp#'
    'UfHlJtEtMeXqQKeX`N0|Gt$w!qXUa)<@g~s6%m{Z^bY*z>9(y(_<fD@=QfQ;{)Cr2-Kn{WWbyOQI09ysQQYHWvp*fVG|Eajb`D&F'
    ')oIg1Q5~OEB{*T>S6Mmlie+<CvRCKWPOG8Il?`34Z0Jj7$9_Y_P%P%OA|22=+M`uWBwOkV40T0zI)gG<{gS)5lIf6bjcE;|?0#sI'
    'bX@KY=SsIocy9b_8S{gUl?S!Pm?bzfW4_I@UebskU{nW8g=As|x7IatF2lTJx(=G<5lz0kyyZJnWFq}<spQCIqYZ4-jRMHz;Q>W3'
    'k)W<v1jI!t3yA;AaM;Mg#2_ObO9gnCcGHF5Lcti95(4=kKe$^fVCp8i874+M<J{?#YXqHj)o9Y#vu&+>d>OdfIeJ)5pLJHA{ka3$'
    ')x@mB?|h}~F0_J`qStIYfgbN9j#M2*a5Kz39p#y$uRE1wXKXW5cm8>~!FDC|O%FFpD@9zjQpVdbr^hx6dXrtDo1$RA0!SOzRBc?h'
    't0<=)@LUh=v<JU7J5EE+2cKbS{NOnIW3W+VdS^pDLkOEe4L>tZMv|95zdKlBG{`yu&?W)n__X@WcS15<KPZ{jnB9jGm#6t2+r2zJ'
    'F<~WYCYf=m@Tjylxm7YD>sR%i@~d(kJ0%gS@*xGAeHM;RjBBdv<_V{wu6(erU(xa;5gP@*U1!l~hSkDddH(hh`%?2-m!D;QzEoqD'
    '*O!U3htRO~@Ene#>|3kkf+pPfX~;zNr6qh;k-qk$tFmM#<+jqW8B4w1QEs1(e;DiBK~JMW#hG(ohn%Q^64$I!-|BG|RE|7=Z>l5@'
    '=wv-j#le(K6U-BL&o5oGoy35WG}@g+SNZI?EdE_*<UVVChl6XdSVYcYUG(v$arX`szom!>_jDRfqIlTX5nP(QX+c;*Q$p{pSR|CI'
    'H=9~~K14hgtOIWnMTn%3OOd%ZY+Dl?$PNy58pN0lwrNm>$S0a>a~#%!+oYC^(4u=SkX^C@ixoQ~^2zOyEG72-AKh@oMw-Y}uh%jc'
    'wp(Nucf!U1sbfocaEWp$bIfT3ZH~-gu^BNuh?%kgTdr!!YaZSRfs!yBwh<*YZybfm0_(h4t319hT9Dy9DE65<XXIN$Zq2^8QgF?5'
    'a*J1=>_^4&uAR<a`EJwBT$0ybOE7(vP#SyKkz#I_Nn~Wp<&xXvN$lKDu`5wJ0;SE$CiAs;S(}TsK3UGjy2sKwxmDJI@>(#yi<FoT'
    '@?H&nqq!cOi{^Tbkm>xlm;CMJ9llK^QMlq7w+FT}xh`PeR2aQo5OtC58;PRM-YgWQ3{4oEuc7uR{P6NtHAhAmEmT*q5UQ%D85y()'
    'C|OW+Hdc$YXmhoknnLOSDv&k`rPQZcA=N(JT={9a4VT9KS2H6lL8TRaUjF422TS@)Ii!?HFM#0hCLYM*D^VFJIj7?ljw*C6A9cJB'
    '-gp5Qc=!`d;N0PZTdb<G)F0MK`5^GM9Vkkk`v0J_u`aAufB&U0(*sDDg2(ct4;`1mrCn9zcf%MbqKaaWSdvkv2wmP3xWX<wBW}_&'
    '0<Ek=e)Ya-I>jKEX}&39npM;A-BL1C)=SNj-&BlbCXO5eB_Es;P4&-Y{!b<44g>ZAO1*swLscGWmk(yjhjR^QoiWA&vq=`S(t|`K'
    'E>k7~x*3Lz#G~`TzE5Pd8!J3vhK?u9S0|9krjSvhhz>2$3DiZtRw89l?xwX9&HU9=Gj~ch^KX}KW_7}uf3uV`ZuY5L($2WKveVQv'
    'Hf`c8d>4yh94*;38!O$(Qg5<h&YSFix61z$Tg3|H6;E2r25P(EDVp7j2=Ywai5WB+)&B=JoTVnLRb0Pm$5~(YP`TzQ^)rg_UC`b-'
    '$GIm*6`xCKavY^(R1x}ng3l$^Q{n=DJJu@g?HunOzj@i_+(7Jj%eBjS-Dp&0NnW56dBn<SJ;>+Z{Ts60+Ua*$;&o%o_a4#{47$W}'
    'ZdL(k#X+8h39F9SKIA>aqU3doZ&jCEhpVZCXBBDBy>pjMb)uh_2x6S4SX{YRj3Es!{M*YnhbpiTKX+en_<G9seJ5EczRRTIsOR!g'
    's00+8=;_@LRdc&_xbu-NmZqBWB}s`2VpLfzf`YUZ#GAcRBgM}*heyY^4Un^89K76n`vL9!p3|>?CwT@|DLQUM(Q#wq+W~)NB{7_?'
    'vXEdq5Q8Z-=KR1e#H+B-8n8~5(y~;4Iv3XPGE#}Ml=|;|f9}<{ylSo8Xh73e^U96;^y8$4=+dqGCC>enH0qBo=^s(+1)Pp!9;NER'
    ';Xvg~B=phg5g$o(o+LdKy@Se&*o!1JBBJklNZa9VnGNH7B86}WU`pJvvwC4BL?2>RtwuTgtH{^+bk_ECGMvkLwVLL^QPvZV(XZEF'
    'LfEQ5`r%2itY9=vS}GmVBrDgd&S88Hc7;97qiYJmR72I4`vO8W8;tn4H>Mq=C-yH&!h?ell{_h3tOB2j&h{gYgp`?pxn9g?3nDx^'
    'X_<gb;voq%%;P7*y`wwHq|`O#dXf}f(lidX#Z}Hp7!{a6Vpq3Qa^qdIp5eKNUCR<JlUaKDT&u+C>p5ZdQ$@bG_}u>SGlo#f@IsM='
    'jC5SVS*whcSzPf;3Jz&)y7JyJNsNMxg2c+VSM<9jk@AxAeHRI>)~7hm74TcTV&9r|`OLB?f~}b&1Iy;W43ujF(|+-M$kG4w{v+|M'
    '+#{I$3LyJ*@4mhlD6^pj@Pc##qLh~=b;75gE@*G5#NC;s-Srg^HLVw~=lLb-G2UT0v+GG?wRBxyoU-J|h7}Z)>r9KB_aw`m-ebua'
    '84b9k9eyxGLfWKl6`^B!)`R)hMGZ~jHnz%`YF~J7JRBg2l%kH>fRpa{J>+Q)mO*ED5=R%G_%$vy46qC*mtEYZx$6f807khd4AQpU'
    'w5$0*4fE<}rEugd+Qkk0u&RODXQg;7Xh3SI?d~7#*3B8ebNJ@?ADsBgBq&ok#w3f4<Ta=UsS1O$j44w(5!;hdc<LcT-L!vJl3E-N'
    'vB(UO4wZ;UNaA0A0)edFeEb9vOTAgabf2-~y&lR_NmIU;H_Gy^;&;kMt?M35x(;onA1cdgMJ2QRbRQwdpY9{9qow^$=)AJWjiSAu'
    'e*N_`jG^~N_xxE!9NXwNhMw+(Rm_J<5~Y-FzUH;Q<2}I?N~+I~Q5eIZ^{FYi3r6VbY)Ji*bm^&LEW-p;Yea&EOAApd_FqT{oz&P5'
    'z0(;xG!)VYUBUkADh~Kk?LLsF#RZm$2Vmc6wW1XQ9~QO4D@m($n(77v!)a)ds9u4G#41V1a+VGgFPRc#!bHiDjx>e9HOukBVUK;='
    '*sXIXS{liW5TvT6AQPl7cm(E8(%~W{gZDh>*bJofcwV~n9;Gx=wc;<B+asvHeI5{ZiDXhkqNTm4zzsZNXedtdbP8r_BX!1M#$N^b'
    'D8HOOcGc65M}lq|*=^1~n<ud)c@p(5R**tU+P0k{l=b(L(9wZ%Kh#bo#W`#JB#JE7R>27wXAPgsdzZvK`XUka0Oui<z(WKbOo;dP'
    'IX9k^iUi}0V0e>9Lk%XVc<%`tI8SaVn<W=b_M&K_yH$}6ZAm7oh^a|hIZ{QBO#0lx^w!+DL?x$#@P?zI^$ul9V^Rx$x9!?Gi@d4|'
    'b4wXrf_SAko>8a@UMpJZWx3)G4tMh@BoQrR<TwP<*g~BgH&;_aRT|bfQ8kGb5%<#6VUjL~qmeC74bGvL`q)f4RiHJKG-D~$dUk3&'
    '`kBxg(q4d~@kTqwNLJD!KbWW5b6+Y*@p?M6m1`a8UURPfh)Tuq6#d~zJTm?=lDhPfWymq+tx>3A;Z?OHT^Ja{&@L*lfT%>f-)m|>'
    'G9C^BPSlaB4BYnqtZlW#>xNjE8pjSy(nH}4x$uboC73^PRw>FMCE+sJf0;v$uhw9H_ahEMCy8OsDJR8YSxws(c&nRCp-;@WaDd72'
    '>4h1;{ztjV<TEtU644HR=~|yn_(_?4Z~X%0ft4ZI_Wj~g-*g%=mT^302FqqD->bs0w%e=Pk;=hvc2g?!CxkCy8JVm+|5gjIN@t*r'
    'b8?(!(^PU&fb-#O1~)k!w5H<IMAr<ho~6=+TIc-HCEpH~gwI-riGlzOrd?{f+oC~BcME;|l+kU;cY`TsBvqfw2jT+X*;qfuMhvnA'
    'PEm@BG+<^=MQ|==9jtW<3f(2T#J3Grc~Yqsegv|#VI9YumiS&sV=H{DZfnf)!}zY4E@^-l1>k%I1dClr)^#lV3fD4cj_IUKT!f{m'
    'm&pe?X9_I$IvZ+ZMSw+qg3Fdv;w$oVb7qv5<hJ&&Qp(+Ro0h=Xr;-=^EW(i-zQLLmqb*|YDyh$dtW0QocYH^O+HLypRZZ)~lkcQt'
    'gDcQ=JpF_@Da%h)98{4>-p*&sArOarsGnPoAlhddCJre%VSGJ|3LXZ+bc9aMAKhwMk|myl7f6}Fr6967&sHkKxknX*lT6ANB>+Kl'
    ')Ki&ijBE!vM^%*k71(RUEHDehWC&F}iA^fSX==gMEOa_U!%G*)LSHOVvr$RKtQ(dq+TP*xU|011C69zd6adJbMj7W5ZFOU(#dtc4'
    'a*EJK{a>yH{Z%VLAD6tL(t$Q=Mdiwfa)2W8MF7!6tAMpKZTMN8CMPpu5<@WGB_&~dknCgFJhc&KICpEXT@q!#v=VK;m=12rA49nD'
    '(1D;l$H&tFN`?+Khu<i3D1{c3J%6iGf6VgQV-y0L3{QrA-0uycXpF8=8d)sMTu2~JyZht(59AK`rS?NfMxJ;gZU5PbwgQVaKMUi*'
    '74s0o3CrjV<WJ?<kO<rGGcn$eqw*6k>?PCDVh#YDhG3A=uKsLPb3XCJz(aNny@T*(5|w`nC(*f7#TBE+AJ>0)e|oj(abT1TVZXnK'
    'p;JOU&ontHSg~_t8i{F@Pkb!iL$6w?J^rEgxcNh4YwJm^{`g6?o)(x>Md`rHoQ%U0zkGgC@&M2i21zPUx^Gk~)kdS%dfcdPQL~S$'
    '_2w3R5XZ~juUDFlYOC?&$>Rq7Z9VxRD%Yy+7bM8aocx%imj}R&2mO@IKdST-skoM6%uywo$i3t;tFJED&7eO&1f4VhNCAwst?ZmE'
    'THKPVv1@KgzA&c@-KfqwGk&i-nw~-tCJlA+w2f`U+KzjGlunNF1h;hl+g8a_=3^2{Po@i#@0s~((3l&E%b-s<a~P-G_Sq9qPDxIv'
    't`3J**&=<Y?8`a7N+O$|Ely5Qk%MUDD6zHf_=a#8CykCb;+$k$$v5bTuAc3j7*jNgpyDV{92_1rbtF%-Q)IF(2<^!`3p8C<fo9h3'
    'b<s>sk&V^WZ)b_By$ivgTE_7^OO_ThouND<Cb6j%+RwcD*Vnz3x{DX#8h$1PhPjI-!{BpJ*l6!BHHy4ApKqx-c^2NK=Hx|qTXVfT'
    'HP_1!tm{t%if>nm@+2&-M0pl2twc#5^_G0pb3eM)<t3rSs$6&U4XG{3H*f5>TS~7kU-aY)bJKj)OTYT^S1X&=ww%(Y*HYT_+AVFm'
    'xvfibTu<YIvK-*RQ+{#H59-B5QrD}IdiVLs%(hc`-Q~k>%jZx~47NC(1DGn_Tv)d#FSE-u`+UOCx%MlweQ<E3!zY=@!85UMFKZ%%'
    'd!)uNI<jl&C1Z;HorQ=YGPn1M{iP2cQjZ0%X++~!6S%$cyZz-M)amrk&Owxk6pEew^i*9ITH0eg{8hn<UM)$27e$-asWy4hLYxRX'
    'eW+6?#!X|8@{tyRjabl|`Se6`zgc<3-0XLjYd_At>*k4n&1?Zt9sak@|2AN)O<!p8&By%j7XSO?^OC~G)>kcl^9N~M^-*&hu1R;b'
    'hWwF!YL7oVF7r$Bkt{p<c8-qR!>{=0JFt(wMsQb`6Z?d^{JIlLe2pHa=hKqo<W3clFEm@J50lpC)r`Z6lf*p|mYfJqL1X(oAmYAJ'
    'xa&?db^o%lCS||9Vs(zJ7uT!)HiUT9a57YQ4H|rxGhNePjXF)21pj3M{B`7aE#kW->9wPaAbAr9Cz1i7ewl?3P6^Cn$zn3(@`>Va'
    'qb!%c*pQ+755yK_N;;<DDr)1ZaG<nG_{G)iwld)tSAj9rX~BjqPvh<nRpR-%JTxnLc-T<X5V>nh@v`BZ2VFjVcjS$R=aF~xdJmH9'
    'V9M$fGfB<{JXh*L-P`%4C$ihsbV1VgY>^-$TU?*f2Lf^yy~#8lha+x{3nK6rY_IK;h+~AIAh#bbl73opZmF9tr_!X2(uSOvC7NgF'
    '?$MYz8S|2(QwzxQFK!!4rFI8pDQ!p0ok{{cI`aPZ7K!>K>A+Sq&H7Msvq>a6KQwIUR$U2Lf6@a}#6s5fl%Gk5yqvTcoGSphs8cmR'
    'GOprzLnT2>EJ*A$GYbksol~Yj@dV~HSj`pG9j?G)3YPK&d6ocg(L2h0rI0<S|7G^TW`4Qu={h0f1tgHRr079rVgI~)R`;*dlJ<nM'
    'C6}=9y`Tu)_H^;7tYKsh!iHXmq^ZB^MqS;g<Ho$!F-m=2@;2>oC>4;6+^d%ArCO;G6m-nK2PV1UY!vnhgD;hmAD+6vLZ1~6qR}Xc'
    '79v}#<o2Z7acT_K*u*)u4WY{<V>I#W7hFYy(CJ5xVs9f3p0UJImb(!qx3jJnTt6g5^We6z6=t^)BC$MCry&g&$`iumCkadzb!B7`'
    'GLhO>pjd{VAUjs2DR|h2q#9^dm+jNLw*KX9o(a7Z8|v6gNAQnl_VXz#i<)aW^cJXc?&9EUbB}qQ%=sl)ULM$w>u|mq&s~f0_1d>Y'
    '8-EL)n1&mCLxsvZjPNp6csVot4OAViH8dSAQ*0;S;sT=TXyvp@zOLH9)gORkiBuVvnzw;F+`eN?)9&C)p!k=xAa`QGT_zl?4A6GH'
    'OrR@qVr`~;Ib*(zGhd<``Fich+WL_@Xh@1P2r>`Z?Vb#Vm%~AHf@Li!<AKID|J~28ey$w8IAHVMo88@A<jlKLMXuRIq{^r-wfc4P'
    '2^N<8NBt=?nek<vUPUpoud$&31GBt93WtB0x~ylTA2lxJ@{4mKFd}4J-l}>hL@q)#YVHr)^J_ls;?(oi9w(LuoSlyh@UGJW^}pmp'
    '#?n<Ef7!@ocOOWKX*<}1V`bZ%KP_NS^VFs5>?}v{Z;Rsvx#q$|+cqnD#egiVCZ&=+i0$^*%$Aqjn&MDmQamj2lW{Y@q>lB4S?*6;'
    'u+`zabm%P>55<F*8p%Y&Xpfn8>MT<7W+kzAEeVP-Pz%<`bOOkowr&I5mz<Qn!k$<&uC%_2N6f@GJ(3VXE3clFU<{@Fz<%B6f{ng7'
    'rrTa}97>yddcy{;J!ylDWqHv3_1$#?w{Tu2clwIddH?Nxk?W>Fw(**fehsW%GY{6kMQr_@bNl=^g})b$>&wb~IIDyf<2kcViGjbp'
    'H6;cLT#HN$1J^8jv;T4D@SPDiIb=uNuMulTlP;^^oNxqRad=i)@fB$mGl?y*6qx-4e$O>_q^2vl<gm~5q_3oEYO_1(*RU@^LuMvi'
    'j(*JGQ*ojVXxdw<t^?`V1Ic8hK&|0OdlI?!@^{B5ce2ddJ}%41wU2J@^L<)Pqu{FPk}&mb9!}#`kO?h6sux5G6ojC^%D?xj?xssk'
    '5WP#nHAlO>s$(cVbKH{Mv@OMN?a4U-2h4qYtxBc#cq8?Ue6#<d-exAgTqVwl)(=<?^(ac6U55#4AXzmz2GP|y;?krEmYYNzl?PEw'
    'Opp|E9b5IZgN>ZVPj}Ci`q(I%H;7tbm$`Cfm^Kk6b@e0=9|Q=08_O9we6em*K_^YRu)W=CTgBbCbu~L@Hdo1__qFp8EnC|Rsd3v('
    'iszDVMrI*%s6mf><H04Sqvj-dkARzV1P#?6ViG0wceCa$*VXATi+SKZfRL<CY|}p-2lFQ37+O<R(wWcqVUhxJL(6~DPw$RDe`<aH'
    ')GV?$nWJ;8iXM+q^!VK|4$bjASgBc2U(B_PSN!^FUV8dzJpcTO{+}fvma6r@$KCi7NCU}GBmHyL3}R^=+8Dfx4q{MTv=gKC@|*ov'
    'DCP2x4mTL<9<|4%kSQklB>$)yrHCF_B;g2(WeyXF2B+fIZ$cH6GZN>G+i)`kbs;rv+r_?#N=wUFVqq!ToJEL#mlVhB?+KSw_VnwJ'
    '`vR9*oefp<>ud`ZUxH?JwxTL4d9wElT*-`vR^aqEZ>M~MRglu&CkPx}W&{#1!lrj#rp9JEuK}FMOonOHrB3*kG+*0#I3J!O%H!cV'
    'RF%J1OSLbaA66yd8r`Uu8n{tcfqHMf=hk=@Nt6b6qg7;wHsaq!Nc#Z#;|<R-Lc1DQVyb^M#j;qL-%?7__Sa}*Z4KS1x(3g~D%w+{'
    'mu-L)5KMxC=4xNx9B1pcY-UP%E<DxMH0EO9Q)C*3{;ysFhBSh<OyT$Qr0RuL=n!h^FJ0Hwb^S{TLo@3DB%sBtqXF)Z)}e~Z&Cpg>'
    ';cfK)wfE&+Z6n*h|MMx7zN^y`7-AVhXcNdH2^m%h;R2oRbMn$&9%SRjQ!Lv+x;dZy?WrcIWSfLO=f1aY4~Zp}s!BEN+T(BYjaY8n'
    '(2dxX_^NKG5~7@jn-x_mmU=3xR4nyuni5~t4OQYwyN27To-|SEGEwO|qDt2SM9kU~EFFZwD+{}&KtqoyVFnm|k9=2@z6dMqisp`F'
    'K;^`gF&*T5%`1!JHt*D)dSpB7`@H?#5+#bh^LYMu#mCHY>9^1(T+e=AHVK-#+%A5nA%nyGxX<6B^!`T;8+@ntm%@%CW|a18cQ~fR'
    'U}cmV`LH0t*ucl};O+^jH+vP2`6}9-31jEtQgZWt+EiSI{8&SHVoUb7{W_2kZhdwQ?ZM@hx;^jYAIQ-^n5d(CF!71+oCuNV4RLUs'
    '3Xwn@I$rYyGI?+s0G$foVX`Eh(~#|_Y*!`Fj1C!5v4EXWR8<O}F__4%za(!)xf9Es8-1rDcfRU7n{wxB8N+D7P`D&@IvCYk7!r<#'
    'NgyG>v=8)=xW3OaP6sC{MIY!^%iLVsq1(fZQa^z?LMlx1oAbvl3Kqpr7pYLZNglNox?iug5#9ttPwJ$QzW7XLXmb@?SEQ_Vp7R|n'
    ')0|Bzp_&7FHR78WnY)%`uiGE~4eiq-Zy)$cPS2d0H3suR$-ZIa(Kel$wFXaOd9taWR5a_EniVI^uc|+vG>`Osif}CUTJo;UyXd<s'
    'bG=PjDMpmayTJ{Uk@ymQsTe{YfW+m`GpZch9j$-umc0Akq#D}5a<{|&bxD4yxh(0qQ}a#5ao^UEOYdKM9M<{*1}K$+Y=IT&qWH;N'
    '<bl<7uQy*%?~TamJ-Mz(MQ{Yr%X<0A?LEHj+J$>d+9v<TwZR=3&EzBW{ANGH>>zd`nK_v&$xv$F^CvI*aD12}wrr5<u~MWO3<;{6'
    '@G{u)!$ZujT2!t-?M=EP_<_-*md&ZB2D+DK?OL8P{9{$BU!#{64QO71hL0H{Bg}+lb&spt)i-1N=Iyc(rs4u|i3(}bN&WWb;ON!K'
    'N2O|uNJpN!5xr8TiZJJ>ZW@h;4IbH2a5WynopF-F&{-p<*@#A&oQxZ><)<L|MG^R_-burcC5LVhg3o1IB4(yjLec^6m$8oGgdAjQ'
    'RqjXWCys|w+gM%2pXXFi@l;?=4fhUJ^bJ$0PA!?k5{XNgMH|H>{2NAV!lPwna?5D1{eDYY=Yu{qnY7{Br&96#ST1z9c$(&u+QI4R'
    '!O;<FC&<-+nf^?sInKhGV}?IR{xeHC%w!V1<)n&Dt$KE{eOdcVDWb*wQ8{#0#l;Xm_tnXcI1y>*M2+@?&)x|z`XG=DNt_vuRe7Rz'
    'l+0JGdn}=ykN7WaQWSXC@RP7vOToMKe*F)aao|&*uWJv+=fQB4jDgYs6M^}Mv1{m>2p!`*b)2SXMddnT{UDZ5vDF_pg0o?J0>_Ao'
    'JMGoKc7ZZYM=b&@QpgybP=xYg1?jY}Ga$hVptM5M{y?ccSe}7KB>lJ1A(=Ge@pi3IPZJyM<RdzPDdk7}Gbzd~Ey@l`*;^_MSA0OV'
    '5~gJ9dtjG~awlUUvHeVaT9~_OT8vw=xPq3%Wc<e}`MG5y4$3M;yAU8OAFG6@J6h7p<O9%-g7AZelUw;?l>*u>E^}>}Pby~>OJN9Y'
    'v1LI5;zZR1kKhT*HA?M7I-GHtue~zsm)O%lT-ZSDZ6H<+%!L)O&y$KiI~GY6Z4oWnTDE9M#zh3inqo^qa%F<My$U5!3y56g#zwHV'
    '=`qc=hHY9Zn}$&1obsEZHN!Obz7wFr5bpD|XK1|tf(ruMiTsri==;J-;(R5zi|~LV8KbQ|Y1w@@><u@7JB2@Q3SUvaK=x}FU3?Ax'
    'yjkKT@nJywA)OtR1o8G(jysOK_gz>KwR9F9ss&5OE0>TCVa{?`Y`>O(0<SZ7xzL-td>40a&-8y^zv9Av#oyPjjLVvKg)Fq8e}iXl'
    '!Q!NJsDkON{rlMu+e!{8Q6^b1e*`V$+$<kqX;LT#k;MfZQ85Gb1=*{}b%_?g!7&T;e;Or=&Sz+y)+uIx&~}lMQu?4|(plTd#`=@P'
    'o55gzM$!LOjPOw7T)*PT5!4l#iVct?ymtOZk{kdVakGwW1;WE252fEu6>LNhayRkxLx_6pL5P3s!(fy&2cr2)E}iN2*z}v5CzF)x'
    'wU?q_8-Ix&!pTK`4D~}eZ=3TJS_0X}tIEK<c~Hsz>7u<5HW9)PNv=hc?}1ioRm}4ju%J`4iNFJ3lKR7$%I9O8GmropEyZdu{E8Vp'
    'hiv44|76l=Lm>E~6Zu4p=m1yuizkoc$139{?;g?K3hH{AjJxC`(u$Ui*Vt6D`Z(#-Xy1mzuj~?i$A}M80ZoZ`Ro~_rN)XF2^FjG~'
    '^F_Ig|4%LU?9aFq#J6tBR#p^CepZl*nqPGO)=ha8Aw~x<E@^i-AY(ftO1R-fy!w&nC&5(=wCGzS0IH1p)b7<M2K`+*S_F-0tCLL3'
    '7^mdyO4E>5P2Np(icQ8*-opeAOX#*fREv`%;QCYPL=a>NA^2?<2oy6S_KFCuz7W`5${PN7`|8Wa2L8DgKQ4*9MsW96TG8KY_3M*y'
    '_uBB|&S1~3aB!6uLn)ZyB~`?iSLza{Lb(RB!9iS8O7ua${CaSD5_zPFG$9LN2sQ2R;c@c}R>G6VoADGqXs)-nDk$_~u{c+Z9#S!S'
    '!8Lf##@Tvf<&&cvh0s2X#Rx?QFJh1t@J86E!wybIV+pZ?7j%MzRtxALVt4{utetMq1#&{Nq{J^;iw5QfPZgx?0}>-vTOAnAwV<*('
    'yCNqyL<SPUke4PmiDUmeeE}ZFR^lIgpFDyHL51zEsfH`1&Y*NpQX`wqNnjiGrm#r+{iM|e>YTDY9JU3wrWb<t>EwL|{l$&|p<fN5'
    '1iR_1$>B#{I_&Yu`JT5{FVK)ynPL$vPW%LM7y1RyDwx#ZgdicSf;wRumoRh(!0;o;oRQ43O~1zF-J&YvZ>uttg^=-~n8AqDCQ1~P'
    'QNiM&+5($VetdkL>$TJeOP#ji3VT4KSf*>MZqHy^4xs)<;suRV?$(HcQ$9a?&omR)wyZz|)gl)+UP&9*Tb3^#9u(R$Eup<4p`QG9'
    'v=O|56|R>dlecMUMWtZJn|=70%4!GuTyL;52&VL6v6_L+U~X{Xa&AsklgQqSx5LVGty96tsr-Q-4K@6Zv28>jA}03efUshrast1l'
    '{SpYmD7~tWXe(MILZOLS8;-w8yR8V#O{b$#4-tH+zereJ$F6kphas;|FebWh(brAiB&W2_aZQ#hv~=+uX5!am>0_;GVbfz;qQ7I&'
    'Kg7k(-Wzz=+8i?JK7jG6o@#U6%~F_v^~%QMA1QkcXQ^3_`I?g0{J5K?kkw$V4CfSPvhjFq!b+>b8M&v|9UhXQx;Q!2*#%TrPs~}S'
    'F$f3i;+~GWHaw$|D;P~U`Yj|rr4_-E1q4)J8?PtAQ=x~1rnb#tf7C?)ZBCsdXM-%0?%49f#I)*z8%FHbH4=T63^AI4tWRup*{IIi'
    'GUQ<Y#Iw)*e_IN&cW$jcdGPWjaPV4JXq{99Gd>)xH%W?WnKY9*S5ZWPN9~~2-5QrsNr{1bg4;v{>viQ*IJ&?ihR~nk>q|0rtt4%Z'
    'yCc{PL>=fDT9bN2f}m{|Z&B)F=`yg%Nv|cGOqW9n7$_cEtq1rOH`R=K_9?;Ezk4oVySVEc({>wrrs9xHUtn%JogTVqtK3NJ&DX{_'
    'af(mMFGCz$S%dTw7oDV@dPRSQzn&@X4GSDcc4{)ZQw=<O2>LSga))OpM{m!iMRk>H6q!i80`b>m!8-l$%32kekLv6n$1gI?hlAl@'
    'T}i-E@MTAL$QT7Ep#*FgMFo|6clr}$82w06Os3;U)Ry#Hw{55IQoin)zT4j@riSAWuMUr29J-&qThL~vNu{Vdc~{X`;3qHMyi@m&'
    '4*vO%_Xoe}=0pOu`h|B7e|xjz4D7u8X8Kdm=Hh;lJ&AkD1z&&V;cq9Z5d1v(aP$v_VLqoV`l9n+PLH`oyXVdl(Or)`L%S>K^)Lhq'
    'g{X!VJ!8|P3wmr}mP1J=E9O*IoVCd?!5BLRYCm?SEp(oh5fxt{{IP8i%~Hyroi=TXDG%TwoH1IGRg{>ULxMsHY;XL|@I+iLfbdIC'
    'kge7EE6|XcPF5es75J953J`-C5q6+1c7ikY9r?reOK(^X_U3q(oAC-QWcV*TX=JP9BGJJ81%gamhJU6ibUF(>|84DE>ejZmmT8M3'
    '=)T}gmNnKi<xdYo94l1;uv5q(Y?hf*K<BZ)Xh=M@U3zOHmPJ<=d#bz4RVQ4}m}<+x?Q8qSY|^MTl@&`5_I7#4t!*?JldRCFU90s?'
    'x1uojB5_Qf0%3(pMXH$U27UJozneNjKhmr1EoWvjP535)tHAE5%6^|$S++YF5vD8NC+Io8FP?|bD5D3)OW`G|^T?af2+Jm<h`1Hx'
    'eTerCe6ZM7a+=eF<JX5(<G#qJESx;#{T^}PAa2}G-CA!5jC_rf&US0*s6O~yvd&}ZA(M)h@NKfyBb!Rm>*Te;HE_(XPq)WC%Oi`S'
    'GfgOt`0H-M2+74s*6Cc-k~KhldRk7Jx<VDin$4bv%0KbWUk^glm~>#bFeF;1=+KP<m%nK|Kr`($u%WNm4Gd=eYNmNz@FHaV3du`@'
    'UasSM(S)YX>K9W;IR)z}xKni@+*}Xp6g{OYQ~8wmm_9uOS;V%pMpd1h1S8C8jjgW-vdRea$K*n1c9sN0qrX*4n5R*vJ5qVmIgSvK'
    'nDlxb(|R5AdL8H2>#mUpY1bgf#Tg|Bf{)%GL2V^wt;!{HqH&lz_MG2RyqDrs8DxGu9k4r+#MF;j0R^9KZ-}vM2z!3UdTUJmdxJ(;'
    'vlPg<=J&}(w_>u8=x7iNjs4DB%DX2j{ta>0bv?e%bv?e%b?wFx$Py=>E+p~Tx6V<+BAzF}higS}O5#$0IhAxOLYP!oY#`Itdm^8O'
    'P9T`j>c<G#5or;3%Y<=qj3V$s@Wp5E&rX8YkXSODOb{cciCTXM20<pJ0E5VwU>gTJ=trs&O^HIAM7w`&KQ6EDRLVi~rpfZxIQUl_'
    'mmLW5T?Q_(K9m;`!AkS)8g>$BW~@3x${La}J%SaD?p|<L<~hGt)Z4ly6ZRyKLzkM>=XeJ)js?b0SQ~_oDchP770Exp9CQ#yj%bkr'
    'Q=)^rIUV!q)<j2wJ$*XAs#G)Kd<bLKt9Sd7Sc<941rU#ZeQ<*UH7cAbDNgRIFq+VOf=N*OCxIzIo$SB}ZL8~8u5Z053hZnN$||c^'
    ';BHEEu$N5OJ%Hv};$wHUDglLqWTLwivY{ugJc*=hXB&g-$)~GMSKO=XlVQKxJZQBzvm*BK;_|aW{=MqZo`o0mQN=^+ROA|~$PJbf'
    '-Vjhh7|M)doSg~wd{trVh0p@HC4nyaeBZwupYA9R9I}t#HAa$361&Za7mt2=j8;&sQxH>JecYL{o(t(e?#zC@IUcS<Gn8vigtAc&'
    '6T*==zy3~q&1fl@SZlO;g^kS$Dgx487t!*nm<p84y~aU~ZGSM9e6CRld1u^hp?(RI-McozAjHp#><}q}uY%PAe?P?$MPngmc+H+L'
    'T{YC`SnvnxFEP4RjWN$y6*FSWh@9ajps%SwX<Y9WdMn<g{UO^yuOi;vT_n^h7O4U2?$QT2XSRf1%cRSrIwi_qc`e`>57qr-ZAQm+'
    'VTm;y()M0sFN}DZ1?_5(Ny^H4UTSx6q&}$7tfBG_>%+f4!CO$DiC%I<d%cOtEmE==<{F-q2Zf<b&Mu;}7`@}+4HaBEAh#&wC$Ji3'
    '4@%_nZ@XLIt5314>(Po-^QV-x^CGe<h7Pq<hWFFKh2(7l0!(6NDe^n8TQkMh?GU@Vj$c6%7T#PV6Vv?2hXFB)oT?Oq<542i78uGw'
    'ih<??C`ZT@FbunMVL-kU0i^slLx6k>`bYV1hW{u+{_y)J>4&-F!n~Oeg)|wk3PL@z`zQ$4^a_O~1|K%-4yT%moP5j&>$B^2h$?OL'
    ')TMRbO&Kb8BRJPKT`T%e7^hII!is;4B%~opFhMIOvceM>b8Rs(1zsf7$2JF4+hOn0?G0&**eeoGlGd&_0K*n0=2D|t6UY%N9p~l^'
    'L`fx5ExK9JaE}t(zb_0she{H`?=tuiTwkR%db>90GUQ1qI5|H>tiw{!Es;{R6qrzZqDUBQ(6YManQzW~eg5Lj@mr-0wmSKD^<i-n'
    'q0|>;0#rQu5z5Z?r@^^lSSVa9T_s|xCO}j%Y8qTz!sthfra2!TFy{!;l`Z#~`8eFdpd3@-q7ZB!bewwoufGBgRUp^~9K&B<zdU?V'
    '7`1JZ0lI#*;`pO3k3XO-1686zhAu86D;rg)oR`RL5raCWjb5_OhA(thfq5Z035mc@sb$dAuwG3Bv1Ri$5F1kO5l3V3GNwJQSWAjT'
    'BL+#*N(loa%kd}`3#E)z9!$}9gsr;^^{kT5J6sW&BXFo~jw#f{7YU-9>^CZiem6B@J+`RwlcCxtA5%4YXbi?#33Yf48>3Zs)+;6Y'
    'D<;;Ri1&h#LDsn_!d`R9TfXRe^RzIsUKd$DyxwW|@!WPF-?82JZejgu<yhRjhO6c|!zgKySU{?*O6F@Q)RI{Q@dCXCrdIM}#J%!?'
    'CS(Fr$e?!o;n#z==TQNmf*mKvC6|wrD_(~VAK_&km{C9`VVU(+4B(Lc-2Z+@N$2gO{2pK)mIQk6u8G^_ARjbJAer!=vDvd)BravG'
    'Z|I$nN&M`At+R}6yWS{@BBOA{aX0o%Cy>wp#rrfhF?&SJR}`m;!GpQ{>&nTvm2om&9)HQ_6x=Fw3ckBr@SUNN>uEoj4SRJQoA8Gp'
    'U<!X&8~*@@@h6tBK7S&S%1J<Dpe!~kpOq$vqAjU6k$<vwG663x3!AggA{GQ1Y1eGsFRhT!7H!Dp>B*BOw$g90lNMH1xqEFPd<gn*'
    '7WME;zS4z;;qVKOi&bl$@BHs~|BPf-`rBKB?X61~L5x+ZrD^SJn3lvWU4rj&S*e1T+*fQw^0_Y{G7<g}&*MMb#ZkB@zU0Y9)&byq'
    '2E>?xsi8>UL0A+bS<%uXwM1P@4q>Jyf_?B*r|!pocp_f|G(q(preXSLY<^bEPs~Sk=PytV#HNARG!UBxVt)g<Z?Z^@du;ipn+{gn'
    'TPllH>4x3wlb9`_qpny{gg3BH+y68k(`_+*wI!aDkH5w_eXHj%40S$bW61@H1w(5h72jSZ(c>g={q&I+(rQ30msmnhf*wDf(2uyh'
    'wXVZIv@cwcjyOL(nxGQak_>}K6o^FarO9N{(;8FjkmkhBfBWHQaRBA}WdX|f$OD)qV#lE)ulA9GkwaM=XnpCj2Q%vPHg~qrrmA>s'
    '<dP^CElRk^I9~uy4GX7tQx)M@WLO`AH;cC$nW*+jF04JLcSa$PSCa_LxU84#1?x*l0EyghpG_KuO=V}MsuilOO(Tr?iL&8C+X!Vn'
    'B6EUz$RkIA+^(O{9PZ}<$5vYOgPLF~Bx*dWffh~ww4!Uaw>CDO#9DtT!-eK+<D84>uMrM_pa{b9_2!PTL!W${3%hCy`4T$XLy%lH'
    'nM-<ag?JE}uu><Uw3;zZl238U9ZpB}W)gYA$gd_k-Yl*>Sq2Q>OkC(iZN(LgkFO|RX4^AAD7<s@E*5$iY*U#2PBtLdLXF44<bToO'
    'z+5a7>3x3cA;BtGmNKP_^KUV0q}M_}sSgO9y6SEl-Ye_-tR;DMvQrA4RDx?*HiA#r9Rw$a-co|5DuK3d;ZRl(XscmmOXuWj6tc8~'
    '&njQ20II5?Cenh%+w0`{gVv#lBLY>p$p#$+LD&FdgMvRXXcV0STeLr&BpFy~ANfefO>57Ri=qeByy|!@`UJ>D6m`lJb*k}s(4F@j'
    '8!c;mo~bvV#kh>P74Sdsw&P6;Yp0d=YuCWZhx8FBy&#>npkhNEvh+Kg1ydv@D!$12jj_LO8(R4x?jRhHv=`{AB76GA<wxVV%Fqt1'
    'mIF~DB$PQgQDdyV@@Tx7DX|t87zreF#5K*O6Z%%;ISJGiCtF3*z{)9aVc5(Rfol7cP<;@EDn_20vA#*4tV?>xSFTuWBi3p5cM-%v'
    'NBlqFZ-F@IRD1?VJ@Kb_@^^3|#~2uny+4ihtxdcY?-svKc~GeDr}U;749bj>dlE-W0~E|i_o4y-*K`;Yl12<7>|k`#9`wKjZn~jd'
    '6xcT#m%pXKcus@y+y*5;aHTklj%_9I1vDI~C<Xp?JTKJ%lE`JU45;9yl}uV44;dCt`Xf?c473hOrA=OCy98WgyPg>%0`EBvuAXqn'
    'GaA`1bS}y{=%`sj=gE@RN}B8yBhmmFQSwd%{u9^@)FhwlCLyv<w2KbLj^Ba2R&u1&3PU#yXrxg-n`3o+6Pcui)%-%2zI-7UsP1V>'
    'D0`d2ESl^9<Uyjb;@@ZVDftb%70fJP)Q@QS&LE?b^FXr*4RxNWo;dO~>@qXNrg8?04#T#;@**~6D<oviS!3f`z%=PM&DXYd=f?gm'
    'Pw%P6*yS||i2R0rf3<q{HP`Pe`**hAw;6CXkZ$G@0obK?Cj#@?JHx$&OpkyyRpiLlGw7(QN5OHG`-uFJ*!ljL;{G-M)F&UKZ0%uY'
    'Pe<4iv5B^Sg-5!Ak7oKu=E@bV#_~8C7Rv`%i+Mi}mYUO?e>+-6YQ$L|PyFxFbr*sFz5^nKU(1RL(xWJ#42=mPah_ux@z?qu7{GM$'
    'd7cW$0`4hL^*CsS<o1Fg=e^-1Yy2Z$+v4ubCdsF;D;cGNcpe&I7J-|C3n=#A?;}s{B_98ir+dl)<-4K3H4pLr;Ord^yzbOrf6bAU'
    't$-KuTy^@`l|FXGNd`Somq!?nz71=55kQyaJokrA3umE!UbKD3;*1;%bq0)QcAgBVCzBi_sM^5<nVPgWqK<AD{jmg^r-PHbvj8V|'
    'd8FI|9OXsW$p$_7DUA&*7=lv9&qE~TE&F|f?kd7n_9w}bXa$U^t&0C^UHSg#Sel8-N+rSG6w)%D4s<d|M59Yu!HFG<4v#h~$afP6'
    '6+uHn3j@_EDU3)ouQ!IX1Rbt-fR4bIRUU7bg6>ALfjuF#s}Y4rnf3{;ir}>{-24f9y_K)aPmstk>N=)jrywD0LZ`bpq4*NYb;$vu'
    'jbw6_pvxR3cjkDX5W{dH*c}tWeX?3cYCBHwb5r)RA-}yaP1KP|5&Bkam=z0-Vat2L{Y5Zafyz08)1+Kbs+|Wkjk`9aHf=M#N=iYW'
    'c1vd25JhNSTrSGD?A<bY^>bSSl#Kyb3Vxf{E@AHKWo(14bD!kJi*EKsT=eeC+2iwe#ZrBt(kG>O$L2-Syfhw9C>jF$#epigL~jC>'
    '-p8}>0;uDP)LKevVr`N*SfQUj5G0J%fzIv<Dz#ZvQV}jJ9TaA5GU|t&l6-gtyAYwsR~#wStP(XpW~^8D5N}PmFadmX@ZvPa+pb{F'
    '#k_+^i8>`1vU+Nz_ld0ekjiuR_!drV>!NEG4t<rYakkbwWoe>?VQ|)aL)d{l>E`AIGL*OeO4J_W(NI;Y@3*?ED`-)ZA+_uPvp+*%'
    'In0gJ8>;2kqMn4*A%9u*xdGw<Qq&66zl#^-@!vAfRmV+F5&n)|30ve<zJtk@R|!pTmmbFa@S>Z2sPPyD{H|RGPTFgE<*%i_?aJp+'
    '-|SvJP(-3ugO%?KN1rp`&v+>J)z`r3gu}Hj9}q34`d|D{Rkg$8Oao#X<MHNEitJODAQQ()RUagsadO3*y7K!DY&PV_XBc{BZ*tfx'
    'v^}j`Q_K#V&~8wy^(1k;roMe?TWD+R3i^PPEms+vg`!9$aM}W1|L(ou!Z;4HRGNlbM|D_Kn1(y21W~yRhLQxJDTxd)#-bF|sS>##'
    ')BTICRR@khf!x4WAr)fQ3J5yViqO%{u+jZs?+^3H`_`=r--4a3Z*7`eW@yPIYE?<sapu~pjIOPkT*qL?3PxZ@Xr)%5GcE}l*d%CR'
    'NRT22^qh%-^K>4G8q6b6gJKeO>lMwmypnZ6Bqj=fNR%+ll5A;CtY!a(Vr8`>fE6p7F4>0Vi&3dkShs9?Ji4nK%^7P}*ls}hWsdj!'
    'NpH<e)Tp+qT3@m)A2H9RLU-#V2$5MX83%|g*18zCT4ZsHf@?O$@k@eRPi8qri3c`(dy8!^$j^ws$*^s2Mr*;L?gAr+Z4)@}x?L9$'
    '%k@kbkMyoqDV>ypVK8KOYE(*$lWFQkpjo`~yNY6!8TdyXQp<qxIK1*VBl85o1GQR^&-OcaLGSGIrapy}1cd|a{ilbYLp_JAmaq3_'
    '>_2{DVljn5Lo{=cYm*TOtH^maFEC_t8M*UPWx&4F@Y<Jt(Ni2qQq0q$ak4KjU2$_%|8#jNF}RE`-VTv&!vO%S_l$F7q|H*W4&)>P'
    '(r!PD=RMq-_wcd%F!FXPF6vapybZ@nrs~ak54Yz%eBwUzcdAm<shxS>`}5pktjvA7xy)E?7IiBw&->&!p8Irb?$gI}#>+CT*<nDi'
    'S3);DEOZ1K9j!56VNi8-1t{dO=(M!PAZs?I4X#b6C(kX>{rvFkoI=F6p76hnUR!>`|4!$aV23kuxgDugm}^io`PGh8F3ilGO7704'
    'zO#tC#uwsTGjXn&IM+;Unn5LN`Ff|!|4y@cy^5p;_|ON4P7-BFk><%2Li%*1_-{Ex#sqP2-R;*)l7U=U?-qd$S;3U-ABld9t>6Z8'
    'qu4BDwpFRU-5qq(3+5wnnIx3E#iq}!-@(>SIMF&C^T40CQsxg}iKVV+!_4nRR;hB4RjL$H^QDz3>YodXUB9@CQbku_vGo^sQ?BH<'
    'wCRzSZZ0^sS}P0Oc1<x<V|t1ei6$3CqRE9%H2GffWSRN;HwqQ^AXvCVvruGMEH-$_xX+Wqa$vn&%7OI`F07YxTv+cZkmbI;d2<Q3'
    '(J-s|Mac%h@t%3#RAoR8QH40`rrkzQYcYsV56JXv6REk;@!t4&d+IzfHj4wpZO{BC744Z1(^aBnSqkVxD=A34OotY}8k<*<T=`tE'
    '?%Wj*CLem1Mc>l5g1R`gV9qe-!{Qbs*s&!5lt~;PLRT7CAPM|eD`{hnio@UjJG?FhUm_xk*JdG+cs6G90{p20GY5D`PhiQSZDEMW'
    '1IAU?R(bQf#xod>^%G>i=!6ZA>&L;Ppi>_=&}Pjt(aHjEVb11={`yzll-uuiI?^cX_)+kjDgWTJgR?UbK<iE70|-2M9VF+;blfm8'
    'oEOtUCkKENyEI^T*LAC4;~B6G38hP+5PD@wBS<%E1a_?+1Ir2x_p%IyPXfdF@JnDNO4VXNk2BBX%=4|x^GfFV<IMBT$O+ufOJerR'
    't|XT4+-Bx^wsYH==h@CZ$vpRUu2R&wodqNJ^MaAf$YuH7WkfT7_cn{VmyygoGaD}$y^MV3>EPpe<Ch5%#&zu=DrvW!KQmImgE0wh'
    'nW90S^$%?ss>+7si{TI`XPpeLKt52Si)}-)7V(nnWZdkgu0@u}%5XqbhvV|2Jo|S~!rUBjPIjy<M9DtM-m2tpEg@^kI4>b?<t8D>'
    '2@g5Dh>q?+Yc^7&q~(>NeNr2~<4SG$*0N$8wNtkB`&ukxvth~VkCMkS)!H%_ivaeYicTUqjI?kg#1Cm)r0?oyJN)ct=LO#Lw6&wW'
    '6eXRo<d96A^7f`v=QiJb;_F1Ly6yHmmX61decf16_F7u{x=;D*ehfrr4_GST@KV(5I^U8NhHuX3x>;tJA}$3g(F=f4oA#$YT$hyB'
    'YnV4y7DVD_W*-Hbt0e`RI}0{fb0v~*6>!|QZ|7J4pAvNMSJYir*j+~44HN@xS>Am*5uXgK%)Lm`fa7I0m3@Z(YdQe#8;|RP16h{0'
    '5fagzUjgQwFyN?Cr6q%Oh~cf0&?g-Z(Sn2K7$B=TdRhoB;cUIEw0itEimSZWd4HGl>aPXTY@tM3JVm(%m(l%v2fD`h?%dnExg>5a'
    'R8$}%l6>|Fx;e4mFZdTo#g<TfdFe{*By>9UX0Q$BRy0q%=9)d<l9jE!kv>b{0~<3b$#Xg`7LCy%+l+(svOD778bTt%{6gyu*m|xH'
    '#{=6~Os+<PO1n-8yd0q6ZMdl|;CJqKB@6Y#4VQCPYSW&-sRiVsh-%FN0kjRddUVNM+zLoAxtGumzz2GF5J*ZW3sVQ_X^^x!NwhNO'
    'G{4jS`IhZ{PkP7u*F4^?EEB2t|0RZyH*g^jZ8)2>KT(uDq8WB|I_202Zc)K^jM^uM+{zrvIbU76nC$s=rHkJyQk3S@q`B;iS=1XO'
    'u)BBcv7yskUv%fbXKTo4P;Xb}QFs%VwIeF_mAjIJ`ymM5cGjD{%8cU!MNd~2?O=Zb_AN=kUIcVhFS8tWq3rsfb||lUMzQ5QBNL<J'
    'ldxM*dF-!TSsN|IRQ7qsi=(K4N?`+A3%=m-a$m4n*ueIJFL*NV3vizL-E5l3OqzSi<f<wR`x-)JCzGpmJ{@h1ctOD5vLIR`FVaAx'
    'TrUNU82>0Y@K?i&Hh_pLYe6O;6M3-KNV;jVk!E5uOP)3?`anc9-VTY$;2L2R>t*;qhW{(@|E8&8{)@6o^Dk#wRXa{q!K5hr8s@v^'
    'KRqD_#@F$&Jf%A(HY|tKxEU=1t@q?N$bihNgySM@*?>w5;Oy7du%=>nME9t|&abU;vbjAC@cXAqKL-noGR^(IhY98MaH?Eh-XPx{'
    '3hMVhqC{O3@DHaRtRPDBr6)^g(^84GuFY5I%q%Z|d4G~m{h_gl-1K@Xq7}c^qXAkr9Q6K}7W|AqSMG`d`Tb|sx21*0|NAx3wXo0Z'
    'Gvfk-!TVzo6|hu1rxeF2^i*BtjmqOTY-G_2?b@0LqI?qeipovSDa^q&z|kpJR(23dM%~`9Gv$)&=u$1s#JOe|V$!oGm0FuXXJcn)'
    'VP0fX%%Bj6G9Qb8tOB15)k~=T=3E8@8ef!l#e5?PJ}R`A1N1oe3di=cjq{xXDb^5aX>*L=*7o|jPJP>uAMUG6gA9it)ZFAxz{6(q'
    'i6LSs=a#X+-mX)gFJ1(@cpa<jn6Jq|*E7#I)pfiO%q*{j{1$fW8vU9$UHC+WOe%$B^4X??Uf^%*!^?{UWyjq9Sh?qp3+(YhAGssw'
    '+89o$e%}OLz`)S0xFG0)GLTELf>0$?tV@A)sJp15m@10rRT5__Q0>B9i#bp?<;QA3urN6Hm=S_*#Jk@Vuwlu?TCI$M2lgDw4b!SU'
    '+vcGImgBrjyh64@zB#8-T_7$}FWqV?34h?hzp2t)ufsn<KN)wDstGk3N_^mTf?=nK29Lf4mEX%0J7}3OD5(C`$!0~;%%oJ`iuh}^'
    'hP5$a<0AlMo}#(nlt|z73lx?l8Rhp(vMP)4F<rxeLF#;qttpDgaa-Y$@z8(q;h2#zLq^8`?Zq2@@V@rqNPCM)#)~wllr}NEmN34N'
    'bwqV{WT)!Zbd}4OOR<nu0>ME@O`yu13pda7ifxYOK2#Al95!_lG<#FRDkI1o+cC7>QjKERr?zJ59#BG>;rNmu=adCoDElaDNAePu'
    'I}vD}8xa6DabMvvNPvn42r#5*`?(a_+>I@|d}lfghoWq63T3x<k0SHFXLuysE<<qDhm^CTt4lml<It1eCeqO<>5|s6OIrPewY-S1'
    'mNSI)JISi%pbSyfjF9{gIt1T3hn6l!Ni8GZo@3E-sJ2X<OWrJF-=1Un)30v&U%mAPtSY)=$F!>Sd9VhEMyM=fciK|q^D%czxy6q#'
    'd>(<8jtHRC0)j{F-_?-WG07c4B$Qx%tD5qW7?z~UjUp{^+E7I~ZQc(Fiw5EH`=04}99<|03w`~vu)+jJ$r<L>)<ze05<q;BsDO8r'
    'Ea-qf`v2|aA&XHk${Yuyi}`|s#rYhce)u)R{8)+)T%BW55|kKM1e0RLc?}2xiO69RDJMrneG^ELJ_EhB^Dm+Rum`WxTq-PBgo*R*'
    '*B&s`%1Z)%hRnk_3Po9{aBksn6)|+}rTDToRAIQhxU!95%!O{ku8BlP{$V&@H5f>u+LDC&1<l~;`aV8{?_Onik)2gk0#BB(Hi$6a'
    'RE%pgTFxPUOK8m&6C3z@Z7<l$`d6&_cO$}mJv@2-31<v9o$^kV&LavNc7N-07S+qb_Tq}d;io<jBPjYwgQGNw2$cf$h&Z;~;{|~='
    '*2|Uob#LY#dz_TL-mQcBWIC?*7SHR>DzTq3xy>b}bpcZ?G_bN7P<{aaELHNev}19?(3v@57?ogzese&Po&~EdBv@wFDj{r)67G)7'
    '`>WI&yWnWHaSiRsaI2A)Nq~np`G=Z8Ul^;vc~d*WaGV_Bmr}Gd^;%t5siSxbBguezeW<*1_!?m~YwZ2@^k_dEl>#WzC=q;ZDHzl+'
    'q!*_BVmeSq6wCx#nH4Xu&PFCl#_NfUr-UO=qMV{CYaDc(bp@kgI$0-EDRJ%$=mY0d9nGohJv;^|N)?0e5jCaWyhuD0U@2<9B(48O'
    'N^ik1;_;ZJ?n~45d!^mNNg#ea0a{^NG6LldyVsO#qcMVRjy48(vpF2yAm@GrzyFBzF?A6YW}wjr!2-r_c_Mxn0uC#!HropMi@fyN'
    '(7}5kzxP0K3$x^%9=_aiaI)j=bq+HXNt(S?amO4f@j4j7PO4L4YgEbY@Cyik4jdpUBME()NJxo>=69T{^r}8m!enYCcO-owy~cV='
    '&fT3fr-V+zM?f+|gdV8rGQq(A3Ee13N-T31uS1IM-$7!$MMR9v*K4qDH#Va@Tr0k_N43yKyo(?)!Ie?TV6Grbgz1<{K?N$^*uX#6'
    'VumUVCu^8y9(Z5OlUW~62Fds*_&nl<5pfK*UIjb-dmL?S_%TMxBx?AmKG_$GXue<P)v#)_m!s6!i2aW}wX&vmz!RA!O7ROHEW{kg'
    'Is4G$C(Cv$rH;WX&+v<@&`}t-*xZZpUhMy<aoMD{zaEY~4&!N=qX)+E!jf^xN!@6FCLAQP+A6ftd8S3wVPH6!VVG_8`BfT{(0MGc'
    'Ff+Z5Nnd+ObCQ<qR`CGi?S}`NoNX&!iiBh+#Q6$&uwugIk3?T*_C!erbBqIw7CyPe@@VYi%SKtiw&gM=pzWwYlD)GWPCc(!J-4{1'
    'Qo%xBz}MWa<arouro9Pl6COo9I*67zfl8XUaZEhQtK$9X>zB`8vk~)&hKX+_jcEt<n7GU-K08{`ScC5mPhTHa&|D4t2WBh>?x)`C'
    'ekI}w6APquoiWwe%`@HQsc(J(2B)VeFu~+fp|TZghSeF<CdN6yLaJlPh!{}>tBt!BivF}a>rQU8rV!rWz->@MhuIguoF5&WouQ&L'
    '>DNaWLqaa0IwMiSQm#=lHgS{W5Sad?F=@#84?6Y}XO&D>a&RXwhzjq72WQuUL{jR=QGIeDydOrdBL+gW;!5trR)X94K^UrBJj^ES'
    'C<l4G{o=!illVnMn4s@#confe_JMjy?(Sx$Y*P|0+WEIJbv&JfvqUn;Q=erDkerh%A)zb`vQwyH<g7`HR+Lj@NP~g`HhvYEWkb^y'
    '>M=ioFX@b?wyTwg)%Gpd?d)>(EfM%SMC9THOc2O-99pf<rYuDEcwzTdqvPbu6l0;fsoo7L9eusm!x6$@7<`G9?08f3-)suxXi5Am'
    'Mpv>xo3rz3x<obytSTM#Nc|*XK%sP}6kN{efQkgM2leBac#{x*a1C(p>Ghw)0wAOuc^cJ@v&l7XV%qCAwMCP2yF2iqyqjA?49>(('
    'a!E<Cqjm>ZQXh9G7yV=ci;9fNm@*39Oj4437*$7Ml}abmA$6qHMSpbU<6ut<=$l!+gu)YLQ%~a8V&W{yUObpYl!@eR>v}_c4+zDf'
    '-b7MQi|&`>!;iFL-hX&`$e{4Ba%c^YHl~AQhFFXYJ<ReRIuk)5N+N@Zr4=UAHJ&Cfs@^lJ031(k_^w*lR7dh7gW!rNsUoxZ4YD6$'
    'S^WX0sD<TPj?`&DA4QemuS|+wnn4sWVb&aI*|*0p-+rtThsx#@Gv<usVvRYktZ+2B2HI;PWz+6}09TW~<n^MdA9aBNzCt843+MBO'
    'P2c2FfB;j~fO3{##BZfs4w}gzonm&iDO-^G*5NUN^BO)%d1jEC2J_XU@vt+l_c?vdisMyCUsveTLE}{7-={Xv@<hzXH0@Z?+2Qfo'
    'p&}-9QHdO}1enG&S*Lhr3>$WHAJG{@bBTs1A-b?%F;owqqv?#2<|09XFK_qIHJvuQ>~(}j>D4>CZoVKs%glb~bZXrZd0@l7s$=+c'
    'qN=9jJGq{W>x#sWd7Q`ahMl{KNkOz>8Z%#Al}73vvMSOM-yNvzMeMl|BET3ND1@#`{o&dzas=d3ojw}C6xQ`B*DMlwlYvgQ9zVh5'
    '-5Bkff;`??CsLUhen`$(PBEgn?o#NrHiSjkfMaT-(;GJGy^VCU68s1p<)<Ix&%6I;K<i}>)(F*xu+~5Heh9Ir^qY;AKX_dRNm<Vb'
    '*A5gmJ_TV?jyO4at;Iq^QwP#7sBT1>i45={CiQU-Hzd$uq3D{GQu8A%x!;Quzah=pUtDjsoUbkgKUVB7hrX)swx*^r!rcRpSu*g9'
    '1<nkrr`BXF_C#l9%cQEpPmK||_!wcv?Rz@ekdMsH)u$iPO!#BWJ+A86;m+}D_?-dCb3M#?Sra<B>BAY&klMM`DFZTKQTp#$RkMXF'
    'N{u9i5z8XX1rk$7U%WYd@y@<ms}1UdVcG*i8E$VC2G&>%iB0>6h8~=fV$IG~b~UtV7VP96{562DN2CT=mc9BE)-A@-jgfcXlVoqb'
    'Ari7=G(@eZk90&?dTF)kBb*=3-yEK5(iu(%c&31C2h(6;Qn)iL4>YFzQ7V3B)C*uV2E)d%bt8z=b<<*wfD&2tkWL1n=6yM6DV~W_'
    'T1Gjdkc{$_iAZ15rF&tRi2ndc$d$^@4y+8!t`a%4@GiurFas`g?a}?yismBvK&uW3#GVZ7L8}#lY)3DA6A`h8jgrw+dGvVLG6Y7y'
    'WeypFe;{w<|A7>uxqv+Eo1yvJ>4W(^Gjq<_P_oxzWG_ZNc$uNodKhFLX{JyKgBK2t=Y5Q%dy4r-b<vpmMpUyR$a+41jYO0_W<LmH'
    '^r^3uP3Subww&rQC4`#cm~$9(kjza-Hq6FFeSpa`SL`%*FjDRC2NYVF1xzQ?=4F*rF9zX{trGnQ-5v!$wloogpCWkt*pl2R@B^pn'
    '#}<rKDKG+)N7t_(-dF{{mT0<X#v&#;K@h&rUTv8HGzPtbWw_8pXGChppP)uj;zkr)4Nl>}sp*^}EHjn3;978+s{tv$PJbp2wZ@dz'
    's1tjMZ|}<0Q>tEFt?fZ4gGYgZ33dR)>DZw82toxN&Z1NPxEQ11PHN@NM-CpIylyPhj>oDmUjGa%^b8AOB6-xime?w%?*r+4jQ8^q'
    'Hd_HmSy#Gj-b27Dy1EZL?8_x)q!WW$!`5DoHhQg^JRWVlQMqT}=jAL?ChL$aSE%5yjBW6f<_7b!t}H;Mw=#*rXUczW!sg)S(+*Pn'
    'Kb2j3{vWsQ%VMsC2kJiDs4---n2T{={81uJ?TEGn3|=X?VZE)_AFVq;^E*|&PqxfwuscxJ@3xY5I3u@4d7UlXZlBUr_2|ygnt#G*'
    'khAq@yc5#TRqRcKi70SM%PsuN1di~d`8JBwwoxqkLglV6Y<Wj>jOWNW{W1np2^Fovs)8YXgo%^MH6TvbAGb4a0x_D^8uWg=U0LN!'
    'parWvQ+7Xmh?_)L$opnEv#%*fW<V}a^KxiD9L~NaZrAJUR0%p0vLxxAR>V!CQ*ogSayG8C+v$g$3FQtX1DCpSW4+~$0_H!(#^^e&'
    '==-(t0G1VQbXrF+HLGOll6%Tb!uM^X*6@ng<=mX3562a6)<4wo%t>B2+89nJ1!Rh86Ox^18G3})zggDA^T4Q&R3sCtDNJBfM!@nO'
    'bU~K)GFl*I!9W*jB^;3^Pm5<ED-Ey=>RozCn?)Al8aSijz?XzO%a~hC6K92IHZRWbOw6m3LoI3896q>M+P9vW+B0Q*sSA(?cx<cH'
    '|6B>Yoa$m($&r*?ISCCO(SKB%;+2&$9JeJmoo$FN%`ZDCbD?Lo8Kw$Hk?V=nsM-*8JSG|Y4h2kZaehB##ZXJMiU*{artpwNQ}MV@'
    '>PBG3`YB#fy7o?g*lmTK`XoWJGZ~JPU0&K5H-3quAQxJ+l-jQJ?e3s9XvXL!i=Q^K9Jh#Dm6Nn=Egc=pw=5?(mPZs4%)A%#mULmW'
    'fnIJ*_%PwYg!}qi8I(te<2m8<5eA)uG>c|h*gfs2Vrx0XIn9<yCRf9#m`nW6^M}hbW$3N<{~K2Zhf#AvUUX^tzpT;!vPK!!=s&>`'
    '-R2HvA2(pmb0+JpMhAH%`z5&bDKvTtbS52lTQ9r)4V3lGlN;*^!XFro1%=8&OK0b&Z(kk~J_MpSWSnU1FuV`P=aDZ47$~Yl`J!8i'
    '6UY{j0grUjk~f+Z(&7}bB1d|S1R8d1Nc~xs(ZG18in0HWC-Uve3RLn6y&TvR<mzIWChWR^CRsG$8xH~kNp_h=%e8?5rp2)#2(h~L'
    'D+<j+elseqD4)p%js@H#nKxnJ*u`Ke_5cqs;R>j+ugN$Q-WP2Ir&a<~IE2*{i33TxTgW;SK;P9+bI%(qlcYbQIL@a?SdL1pZox%>'
    '*RT*)YPD7}gU_nfaBmOCP1sxLq7BzGU}VU<q^5=<WI}RMAJ_Zo!(dc0Ou;H!cWR>Awh|e4$qAJDu6-FstsVtg>2=7?I(V4l_vk6Y'
    '=|fKoEX%n#*gHcp0=xp?xQ(1PUqg%bwz4;p`u%m+c?C@v-?uGk+mkhfGHqDw?PeMN$%n^2unibCD{94%8V9MdWCJtKPRgcaSx=QW'
    'Sc{xFJPdL+AJ)OCf-ELcZg(Z`MF3AM*B)G7H~@FDpoz`OIM&f$^@&)TM9*Gz7WT?K>ltI?*^UPp^Vn|S^CM6Oubi!`DDclUeHHPA'
    '+OY$te?L?kK?`Q+te;yE6%1ok7N3PO+<n-(;sQag-dt8>t2d3voDr8_8zx1cjQT6JVGB*!j$G`Nt>i6O$|&(``SAGQ`O%@93_cmJ'
    'pS=&p^}(g&uw&nO^fRl|`C<-&%wj7k6dPJwxOCCo$I>LAWi2DYQ#UjB3>KC^VJY_aGIG9c6UX7Q=;bcI+=Z840~=dm#KxyV`~>I*'
    'RMxdCi-DM6PI#v>Z`A8TgOzLPyBF(Jy}8IF^?E65)Izh$9J@bhsSKD(#=nWEmvRdu&jnl@IJ7&T*-<!0H&#5pJNM>|!xUl1Gi(k$'
    'xbI!x#OzP`CJT2p4p7-ybKl|PqP}^to6G-zC0#s0OL({3*V<NJW%(KmCMD<Ska2NFB{wGbH6n#RQ{FwE9|i9i+!ky0Y_%3fK<%+g'
    'g~Bb|=foD`<L4sbr6dNDYRf5Td7{!gYe0O-3H5cRMKQ`YbH-SXzjsWrwy4dz6_T2Jtq#b8vl`vWeNa|U_wTdUS|2&bSerN1I+j{f'
    'Nn>OTvd}59R!3~DC6%Lk_@ndC7o|Bz!)P@~p|iXqB%~Aip#79lT}DqM{u`P65lnJJEe*TA|K}{l^X<gnp@j9TV86*id=Hw)CG?Q@'
    'tcARzW%!PE;X7J|zg6*h&uULs>3JV2&t(*z-=OaN7G>w(s_ML>qSF{){{!}4rRU7pek~n9A&yesyRO(AE0$Q>09(v?`gl*i7kloB'
    ')Fx+Pf<~gGMPJ%Q*|4e{)M1B>di7>Pc!wwjlx%r2T~6w|H^GgD{&uB6VTXZMtg@p>A9x)Vcg(FRMIG=Mq<E<{(DIemY8eop<MZE2'
    'fqU!y+vDdE%*q4}t0)_XiVmQRRHKXfU;-_$#xEFI%01Fcuoe+Ilx=G_SB&9Yb4B@fzWCdhztzr;KAhtlxVucb<Qnx&L_bllOp42T'
    'uiF{GC&%RqEO-HaWpr_%V4prq$RR$K7<_yIb<n`@u-+-I>^xsjuex8qBK8_*O6X6~o0}XBhl4;Ku8&~Eum?2I^)Xo;3D6Y}>oaIm'
    'f8N93%)9TMLZ~s4j*_kfJ2?d&#blpQj|q*pT-lsP)~`HsY-=*VDzik{GE0owHICat+k{cfdOQvkgl>?IS%Wpgkl--weoctve77CL'
    'bjFfbrh}AcXUN-Y2s_Fu2-k3aZdRUb?@V)W<LcFcu;^T@WW5hVJB<7?Y;&|hNnbV^)86IA2$2yd&5MnribzIAltn?7L?IlQwO}Zc'
    '?7%0%+GG`H+|s6!E+SO%b*12WRMjDyjgkZ$QT!wa(+yg1+>+G>cFM#TKj^5jOT-_Wx!ASi=gv+JUr6ZSU%`$z^I{m^Fh!7OHqKmN'
    '_B)&u&cw+*Bb)n(5^BOi02*Em&R!qY2c4hN;MKuqfVoH#(#c|#s!e>MHmNsAB7&-FlOg;E1wY}%2<Ti5zOUOv<4Lt&zpkZlyO#iM'
    '`V1?s(}VgSrQ^w?Ui~H+r;l(G4v?ZhvI1d&B&gdp>BYwA21zI8U>H)e?OH#X)LZpQy&ANVCSox@3Sf(OlF9lk8F$+^6cLRQLptP7'
    'wZC0p*HoL`6zCD0)4iKOx(lm;|AuZ5Z9oIlMsQB31TbVY0zc7hs2by!(Lgo=DwK|rCZldbRWL?eVqL1M4_c<tbjs0<ydr)|xoR0c'
    '6`I1(7Zm4_GRh7PDbyt<B-r@904_V+4@wz@N%X|}5{Zh`ih<NEpw)$Jc(^p`gPY*pqmPI-H%@>nm0RhsJ%MQ?(^iQ?+F?hgXr5Rn'
    'NFY+qPA*U<r@c6+PcYuurHy0TvLtC4Y!uoXHot338??OlbZji}I1lu(D^+4N3zotdS6px-LO5OrE~+XmdX*@qTcj3%1D^NgGt5(n'
    'au`;PY|OfT<`r_qmwivlA3=GL@^uH9SncIPZmoD+(pFdjywZZa81fJ3$zwb({-va6+>5<BEE0Na{oThZnp6=6j1E1t;V>x)m69D5'
    '=w#qvO#&t*xuBr)6dMf?+Np=rDBZ$jc(9#Nm5P0?I#INkyeQ)e#V}E!4x;8}(tsWEvv2^)7^zBe$gUf0ED@m4=JAl}m!KV_m@6y*'
    'CTnt|0TA%(ICmc{OB)+OL3)Fvoz&AQnz7T~@QU#SByQgczAn>ZYD)UysvVmBzyY+emcgV)$v!<H+S$q8b>mv!AH071f?}YXNw*hP'
    'e&4JBEz_%kQMNP{4~bCGU|^F^gUaufa!D7b9wYlyC3^Sv{lUwPH*b~GlxbM(X8)>`*YVkS+9Z<;uTo{UMl1X}0ZW7`N}2<tqvA*|'
    '`IyM8U-<a6)~|QEO&C>*h_BTUSmUA_Y91kiZ=pRPC2^<0Ey@D1?Zj$j_mjM*N6Wl$Yly^zb#tWP<4v&|SnLW(uY;EG8nTZihD1fj'
    'N6x8>G5b}uUK`2Hki*3TkUMyD5m$?#7I25wGp<*OnWfh-G`-urz#W!~FYI^}q(Ht~(;j+q=uJNE4$%ZFfzKGZzJ7&R1k(DXehDv^'
    'AK|Q%S3|&f$X21BQXvW<MMPf_0D~0^&D#chq8dfLQGzC1TtxUT9itW&1ffmPN*Ao<xtlwO&tUcq0vy!1cY8s8_vAE}1c<N~+zB|r'
    'Y9j}iU?g8G<=feB=ybrRu5O|?1X@XdTp^np>gx#3i!wBVi^()p3f_=z2d3!=b*6I4IHciCRM<o<Eo?*9lec=F;Q2Kt^J``ie@^Xu'
    'NSYR$?M53uv}RzHz6FH(fsI}l_%>j11&LKbPT|7s?~{AAo0f$51dND-7Rc6v+yUh#AvQv|1!By*gK4s>(<&)WyBg?ohrgXRl#ok{'
    '6SD|p(b2M>Jy|3LpMp@wElHXk5gX(%t5WaA-NqE*gLQ96Gg1dy==wX6uak*3b^LV<N{ZJHM^=kKP*UEA8WzmKP-d)Xc~P`E;v}3_'
    'CdP+n8)>u0$(LP&iZna&q++}yR`YUr%{djO?c+qsO#78UOf?bI#_B-J({{Ov>^hD-b>#h5SheZ;=;bEpP8gxa3zL2Hayy7+2c_H~'
    'g9I+rT;(FH?qLBFa!I_i6^kh}NRDhFSLiMu3BcGQNL|ejQ%{837iof+{3!B#kGrYejZ88O9I3-r{e}cDIHFzg#oow%kIcnTPksmk'
    'R^L$AtY39wBsJU1!5|k9trpK0PT7mt0Xk)G!G0sdb}8$>WE^vIdrp0|@I1ek4-Za{{!w+Db}WA#LT6-5%+8P;8i{iyz9vR)rzcEK'
    'SyT#wypXiJ*Ma2Ou@3sH@^Byry>c7klU)tmXhZpn@)ev~519GIYp=7Q&NC4jH5Yj_?TJvSKlLcK;LPyw0hUR=nOM72<J}B~gLRwt'
    '30csGO557mbpr$#*JFhs9w(@zDRHyuP?nzM0L7^zV0t^rKmbM8fev>Nwg{x~;b1r!4uGu|Up?g>H|jPoQ{vn>MndXQF0HJB(6L+i'
    '>Q9n`R<(|DEAW3~*G+G9BeLW;nRSQLRKFaV!7z9i+1c#q$vleHzOX0u!7?)|-kDfkjU2|{%q-Cr9Do6lA-sU`QrdZ~O1vu9(mr{y'
    'mQ5|HlCq5gCkq|hrbU?Fq?=j<bTyCt;cR0nZWllPh|TS4w+m@M$~h&zzP9~b`KjqyNc@lL{8KFb2>FC({unG}o{$$x?2S{g@?=QO'
    '#0U$;`nCM(!|98|=Lat%srB%>c@buOUpHN^L4NW1+m}2}T<cTpnq@1gP-l?iR$s8D%6`t{qZyaMOh{J&#DnzV?Bw+D;3bTR^3bz2'
    '+L%lbMgTvc5|vDf)^JDrzLK{XI+gWM#-1n*SY&B@arw?$26~mzoZoR}EWa!=+VLSp*J&BrzaU@=%GP%atuPcN2wz+KKDC^GBUy@S'
    'KQSi`jXts(eXWx1nN6tP(74@X6zrh0i1iu{pmV*R$(&t8nyO5jl$-udmhhlTGhLZu;-ejuv+GsLF|8f-9lvwtoOp9PXYI3JDp#^E'
    'Sw(O4$T+)s$*n9sQgy4o6z%KIH*IYB0%^qE;=0bvwytdIZLRloXYr=4)p-9b-q_fy*jlta`3WlSLVfI2^;+gccl}W>AzW@a-8t*!'
    'YVbPB&4B(*Lf2o;lspyh4&i#(uJbyol5(+0sutE~2A`U%=Z{{Og0Q(3NBA?!N0d;(UcJlUCX*rk6~bfSE>Uf2f!@9X4)MII*dzQF'
    '67Li?AN04AXibYs@7L?%zToWUgl~rupT$pc6s0nG@V*`xR9K^AZ;S7m?g`R5D=Am*sLq|^Iz9X2a#@#nesCm}6|?v3q44X$$(w_t'
    'SNhGD|2S4;8pnc?TXg#g5$f?2(_Qpc_5x)l>DMWmls)3|dPzFbpC6pMU*;OIe?8M3&}qnsdp|$5S$E$dV5k;DOUe{+0<@0CuS~UQ'
    '6-J~&XDJ=+TEarr^}A6HlSsg!Q$&aMn%@zt$cc8R%;jV?8=we)$u2*olh0WXg5wiAVW0aBsTX`Ny~1=thzUmH``bzF;Pmw1=m>6R'
    'y$GO8?$#r3if(F~$OH8t^w?hQ(SK^4z~uZm9R!Ybs5@x(YCYx)2kHk2gr=MxUxIpv@rA$9Q2m^Kt=Xr1)@89#R{E(-V3dY=wNDM~'
    'eS@pchk_bg)*Rc|_nWQopMlp_B?2>z4E&t~vXIe)r-Mb2uR{uCBzr%ml!d>aItk{SxH5w;KOK2o=&;R_W$71{T4v1g&>KeFL71_L'
    '4~9bz{n<uy+W*v0IUI4(tkJ9LExMz-Rn(_V%jroiGrJaEZ#mUUUfk#Y*p5+Hj%hN9KhJ}Go67XjEUQi#mQsnE13^&xN1@@LatV}!'
    '&>*X_ebAf`(Tc5H!ow$n_EF$f;@WA+E$Hy|yH`3V%MS81?0WY49g+z3JNB&O?XB7*IPVDKfL9TZ2z>Zw++SSxK50&Mkr0Pvoey~~'
    'P_>|y#PphWG}o$Mv>20BZ1$1iu|Yg&CZ1PZ2N`NI^P;;tb%!$md{bZnX8rmIoA~tbGwqdG6s+%4Z`~wIyuoV?CRrC<`X%4&W<FxY'
    'N=SvDQps$>CKq_5#oGQFZUCy6K1n&?JAEMV4V(&i>oXyEwDfSpmWIAd7g3yH^PKb)OXhld>s(26Fe?0KM2Bh(Re4GFg!D3YlSSRy'
    '7d*!_sv-P#3i4#iv!oOGaNiUu#{<Ik4wP|qs0Dm9HUj5tBa5~e2z|jVEAK#g>I2*vz|Z#wzrokxA)T2a)XjB14J5&0W7qeHb0h<W'
    'hxSXl0a!fxxA!%2qr@SGev}-_SM=)I+Q2tXXB&f*hSPcq3`G;IQqU#$CThQ-!QCPRN1|<RDyr^8iK7JiEopKZlJUkN*i3CpfsC=b'
    '0p9~8cb8ryJKysvPif?LEqnFL5n2T-%AN<T!@6p1MG-uQ{P$5oD=_uC*~@;(ikDEfZ9-`uQTRP2Jvf_`(dS!erxi@y&bXnyZs%1J'
    'iqy9AbhC+x($CSc{dJpYtEWuZY-^e-CG8W)CZ)VuX5H}w1t_4JsHV4~jkRAJ0Sl;dYzz!#v8)A($V4bKA`$a2B{OBgq-j}G%Roje'
    'dxq&1)lb>69i+hF#GVv+U(*MvpQuSYDy}cNdwogU%PTgpYn#Xv)@se(*RYR{ag8K%GttU1;T`I|t2)TAD6w|i-2}J|cUa3l(5`M8'
    'xXhEgG8@=*H<@3?u?5oFA|eLA*6~~PInhn+f{8AeP{G71nB;^F3Mw^{bXD~4%vZ57$f7&@&<~QpEz%CW72AqCQ&fi|-#BbD0rKS0'
    'B1GBj)X9$Dr>}%-T5(u+CQGSV6y1VF(H(fUuqZuhwb5|Y4B=hOA`wqf)4@H#V<4VJ&Zg^<%3Bmad}`ST79TF8?U9VbRMW1vfOT%Q'
    'OZ0WvBL3|!Zyi`|wH~r>*j&A$&>8MjA6HnF{(4w=@;SaktnE9*+P+3vDRA<N6TFf~;uL@59Mas))d=vlIMTCA1FRay^>Z>r7uVpZ'
    ';@ircysO<#@rzt8%aJ7;-YrHl9_+Akgwx?KgM%$ax`W32omwuCO&dkRX=A>0YJEc4T{;b32lzfzFF`=oQgNUAYHS)G6Y?{OxkKn9'
    'jFP-hr-J9w+oQt_hffvl!&@#Rmvya4%C2WkqfGX0<D`mC_+{k1M;pD7%e!W)iWgVK-C7rReqoh8>|H|5KbEBE&<bA3DG*mAb0H$h'
    'u=_d&Iyp-<g8<$uo8uCWqh6fFxzOJD5~O^@f#tg@q<pktIhf-&Sv8V#-{UBX=D{)I{~l)vzc~Bq<n%)gCi?wJxK(yzFqqpa6I8yE'
    'MlB059xK`GOcF_Q!ydH)hC?d7=6IN<<dOq-SqkSoDCz(TzmTG^JHYs*?kJ)3swk)TdSV|Ym)dIZuP4EjvLLdRd58VmNh{1|Y4qS}'
    'PVp0<DRo+-4w;>F=~%<gvv#@*WO9s=RpLz`X_-jn&L3sxWf^16C(guIAB`yP27$i$L+KkS8Iu_Za&Dy7+Q}8-_a)q!G+OcT&P%e+'
    'nK^>D&dZ9~UBv5ZR<{T<sN}*9$&`n-ni`4GC>eMA1XX3cGFx33;|Y3j29y(2d~C^zz@`J%-lKP>Q^FY$?vrTb!>oEdd~F;SaF9aE'
    '8T*>^6-L32(EDo8R0i%Q`u!w%%%Y3{W?3M_^gysHY>g0y&6IWmoRwmUOyoMr;GRm+tSBHcim3Qir!Q5J69-1@bITHeF0K6V>g>_A'
    '4lA?UN2dnNOgJ7V<h{&OMF1fnM=BI59N_J(^#-Q=lFU*;6Ad*PU+}7IsW@oF+X1S|gSAcAY(N^jmq`zIbqghp8>lq0y#?6z@j%-?'
    '>?xY>iDHEi5e{0D8BByQcvPXCY%QtD&^V-=y)S8N5VI2~LtBN(1-@Nphe1N~*F{x}xg~0S4BCqNKzkB<g|2v{QH<$7hwrl9#rQn5'
    'IHTL_YfG<0=O(a6Z*bjt*9gw|1-$<T#1Q}D&<jHqyEQ0BPNh&_EwEoi|5S`<JBMHA7g4^I+#AZPHUE}2N1&*P)T6np-2{dMYf{ez'
    '$&Dr#=`)c>@(FJZ2@#Dlhhel0Sp|FIVB?)qqNROu`J%ufOlHI3O*#zCnP<+52@4j^jWM4NReFg7G2g=nsw7XydT#`W8f6Ln4^zov'
    'gFA(sy^Rf4d%~k4TKTtA&)~2d&=-gG0ecMzS`(^q@Z-<kqq7jyney2oi<;c4$S-iCNKx*1<~$2FE0VKS@n4Py200<rj4y`Zca4)w'
    'YA^DZaf%2?Sq}KE;*@ICTlpfJwn9V%FY$G;rK<<lhZKX{EbA{qnpB@}LEg0hzuxCPpgOj!_W*pn?MLI5g!2%&1~sYc?E8r`j`JPm'
    '(v)Zqq6O|l;%A_I3}}0*=<o|zM8n}F=Xg4M{}U&A(oN-jiCnjXj1Q4Jsq9YFR9-~9I_!A&+a(Z<%(rqY`rtLVW@d9$wHx21C_e=e'
    '5RV5|O28?<gInTqDg3g$fI^nZBq2ipV&!0P*^`JNq^-i<EJF96{YJy86{Z%l6X8I&J*kC?k_+AD1%+%brN1v%5-*m;36owgnJeiw'
    'v*hljldzc!F`%6N`Cz7(Ep1ic4;9nQc`sEI1RMx-h!InTVq$V_f}H5+TBlP;#Wh8aErw2w-Frd)qN;ogE>BjN_We#Qw0F7`iRD90'
    '!ix9;gbP&qL@JMXmNJ}e%s3hB3`(q1%m*<XDO=MjLe-oo7O%HIo8zdmeMj9zJP1GW8x2y#bAnRHjFku?YSuK~c@D>p%v@M0Z(*YG'
    '6fqpfq-VG)Bl8pL_@CG2{IPkc?91U8zL{=RxsrlfZe3T)Mcy`k&8Qsw#;4G@#O((a@8N7b!MxeQvS0*4Gu;+C6f7Q3iBUBEjQ?hI'
    'yR7rv9GE-ds&!Y<`pR^bm(@b(yL%--#lDDzbm_q?$|S@4cTkPI)8qeAjsH1RL+gI<sB+aC8H*t)!$!lD{A5_D)t(u19W&;-WMs18'
    '%(F$hcE|bd8+UcwnB%f>H;)ZBKFsH(QRJi{?isW?a7@EyYAbyBRuVZdRiQY2`g}4YwyfalZ6?;qV5zf;<FD-CPmV7sh_F;z5C)Q0'
    'L6r8m+EDztF8-Ne>|8I!toLK&;<4j1P5_cENbz<$cFSS~vs`SDuMuRf)Kleeg>`x|8<}W2Pd);V?<@}g`TK*D*E;a+weqA5#p_lk'
    'iAwE6ma@bYmk6_ZCs}9q5ef&WJNYR!j<lGHXFB3U0VZT&m(?Pq92;E_pt=g?4eF$!CMnU<y8UFQ5*Q3A$V+ME$Jl``t&jSEoKq=w'
    '%L?VEI6IC)P!8y=sEBvTFM~hl`uJ`shu(}!?g%c~BiG2GaJSR69hm00(t1yd<tLaEQT(YhPvGdH`CHj7s#%|B|5U9k_A>mR<NEx~'
    'F3xxKaL)K7pcTi>yAfU{i6G?LE47D_a~$S}I&WC8N~4fQ1=cU;q+BWXg)8)fv)teoxWLWzfGhBR^Lf6_cYO0Xy)E{6)BbL^D|tUR'
    'eL%@Lc`~JB>?abzaWx)JSQksvDAqN+)<gu|+@hr{lX1piB|YS>;v?c;#?wJfQ!?gb`2Mh$IG9J;<c7$}t)$tT+>DZtidCz#dT*K-'
    '2l`2a7lP~X9mFd|S|{p~t9WvK=6ROLbtLbuvV;fx_xist1>BT#L(sjBKa+cbcpvE76~66BirY&_S1blh$;fPS3VAjqsX_DoXbFh)'
    'J*ORtF{Eq@ZaPY8>^0JO&U8dg<EH9h;0>;O>psdv_$<XCPs`b7o4l(X|CtmcEK(;TAcr_`n#UR_;}Fr<<21(^lz~~(Kn%>98GHet'
    'X>%b4bj;5)T=G@MU|#sfcLo*lWYc~(XAm^M29qqK2?L1kCv3rfM8CaWP&8fcz{mE4xj_ilA%Cj7ZH{&afx^nuF&?(2O+3Go`YaI-'
    'IYK7koP9lDx(I{dc6Vi2j+xFopsgIAj-0mylA{p`Nm8o)8+7wXVmsp@a1>48#4eE4BRD^G;3=b$PKxE_H1_B5WBK^8vqNA)e>*-!'
    'SGY{Zgaq;N2GINl5Yr6;o)lbCvEZnT34<Rp<8T7iof(<frr=>ZX;rI;anc#rdm6+Q@nnJRgNZ3ftnh6e?+)(P@fY9N@rvpA-DNme'
    'iu*1JDwm^y+gSQL%H})n3JKzuNxw^a)3C$0w4ny|enQSZIUM5s6~Y*1{yUom3`kz9+ux*f1D)$z^|J|Vmp*EO=toJrl+eMZp!no+'
    ')avo4DUe}2oaxjZZNM}1%!R*fbr-U$-KO%5CCpR$YM0qb`^EtuaY~2nP~q<4jPEO0suGqQ6qKZTb<u1C#6}QU$1`rON_AO%BpqY3'
    'S!*(Pj6d7LDAVg@U)k6cCRLjL9b3@WA(Pz?C$E1wIDHvuB(>nf@gWd=f~w)a*6~i4Rw4V{A=}nU+Kd|=uD(mggI9ydnbt_$mrztt'
    'Dk-+e!?dA)00FVs`M&x{gu^%kEeSi<n(s{;>kUo_8akbU3qOhsJY@!bRPnm&R5dkNV%V%S8PAkcRKK=U4w60sr=o)n2r#O>DmhBv'
    'dmJMRMSmuzBpGl1UjF*va`0)35+zu0*w%I;g82St^bjNef&Eh8FBtDHge3C6SShS>bN_a?u-CD4djMkUXfiIU<1y5}pvaRs&7`B{'
    'n)!KgGjTk(oh82~-dv)k$NP$*2p=Gw(CL+x_|fNh)4o!f=(t8CVy|FXpCyxniD0IMutfTwOZu69@V<Wi@&;&Uw|Pbq_j?k!&nWd{'
    '*e7&0%_be4y+`+7AQH+^k!~^FP})v^<MF5XuB|Uq>E?G-S}>4(WJGOqo40LPQ~mE~ySUo;yXw6y9<IzXqf_MW4Ye)%iXv4};fkyH'
    'KIM2gI2n$~)%gY2fVCg=&~n}~$~}DVL$f&@0pY)aUL7=%?mkD6E<H)c@29;<7nox>6w={1d`M?c2`S;G(RT3xDCq+->*=m_$E856'
    'xcjd?1@;El)ZbNmTw?O{_tf9c5iwa^(iB~pmfBX@ua<%r#b#e8lb4jwHGBy7t^V-c*Y@gq2rNiRJ;dA8u@lv?7wAn?3QRS^YH?Ox'
    't#vF!aAjXWBoQ`ZEaA)wM^Oz7s;r4JCzn#n63j?mhBHJJ&qz`9FB<gA3U0u6b;eiiH@oo_Xs%T%)Jz-mV^wCmQ09jri2|uwxU(+?'
    'Q;aZ)=~eO0wVTigamYqWNA?&r7I{51;8JV*d+cAQ+1W1}0!<s**t8E!OJ6K}vpiPaqCV_P^A1~L7kH{WiQU4&(lJXJMM0*5J&IN}'
    'p<7r=o!to~C(=V33R~%0*3bY@nPyM8DK(eRfI+Mx4w62#%@{m@()gD%Mm%!1Ey!MK>;wm|_lFi9aUt0G`HMHd9KVYgsa~NNWBO<U'
    'vuB{)Iy0TPiW1`xEz8+}C8xb^i(&0iiwKOUs2W*U7;Crvd}z%jBIY~FNs=b}-N92`Xg_o@cb&RRE+DT7^Fe*qNwfr9Cgv5JSBDGa'
    '#-o<mh88q2Wg3Ddj~^DPMS`b*JLjJ*ieh08mFzSVf~zQ55`fmMOY#B%h13w8yH;l3fO0x?6SnLb2jb{%OHAQus0{|=VEr~D1YdgO'
    'ZonJIVc*(P_?de=A`fs#d?OO55yQgI_yEcj!@UH@UcqpxY0`qCOHbUdx6?dRvIiC!F|8Z>n6s#dHYH4qgZnE(w@VVL4-@ZzR$dLK'
    'P!>-FJP>e`ngpjn6wwSoB$&ThC<;rGG&E{7@N;-X_Jox{U9(dKu%YHf<e7IZ3>Ez@F$&pt(~<Qqlqh^R_P!iDZ)cmn^C`uZ3td*M'
    'w0ksJ3xrzsB%Co5Y^e7G!*=QoiB=U+EX)<DHHa1p<!Teb#`)_nqsU?VuB1?Cb^E;c9${h-3R|f6`y~6Zb_k{UF0u>)DtPkpsLCGM'
    'w1hC31iQbmJyCE(n+Z}_)tKychB8i4hX61iLON%xu$7Dm2YcAtp!FygHq`GT$<`Rw$QK>adVet-w7Z=e!Wy9U>`&#V((Y`CsYK$w'
    '%zZMO@B>th8HOXr&Kdi!!%bSL(XhUj$P@jf|NLgci1$pb9H}xVQ#!gLxXHO4?8Fy%0G_>%bW1WPdonq@nX%B^BeJVwkWLIAI;gui'
    'M24OOTzF9byg|N5p6W!5d5NKM7-+SdeQ{RrZ8UWuWUT@fQC3{)hQ!hZg}&N_*475~Ebx-Gu9x_d6KjRD9`wv!-`?Jg=D#C?-MoI{'
    ')M1q=rP-T|&C(f6`v>wsy<&R(j<(3*b@&j9mT1S6f-g}nG7ld-efpG1{GAswiW$6|_V@Qgsh+aki6ZjFFTi4h@DYTohXb_coWo9U'
    'p-%F}FB4c0!}0t2XcdGZbM>-5sh`8^D8Q&RiY3x$LODKV`2l()t}X);?s?0(MTr`{x2$fx#Q+R4b1l`R_2wC8mvd@6s=&(V4{;4R'
    '_g`Ts<A)E=G0GQ46L$C~Nku%os$t-D`Z=KZ>sdFsVv&pTKcU9`ngg<t4(Kp#c#Xky-hE_X{*rknS?^XvUuX4#T0^Hi8g)vgSF%`H'
    '?vZoNedEm}^*Qq$By@#Kl~sFM5a0)D8*+o~fm=`nU5uR=9|DEHIT^wn{r40kf>T~zUtB>sQ!@OaIXzb&XZ7ocQt+@nU`hFKfJ&E_'
    'NxKg9!Tm2nq|ZN`zIemV&S9GvI<1^6s5O`l>a%*chxIAfU;X*HBn8ee?L!mV^~#<QBG+aC)7i^TEq{jQcBE$lQAnNJ>20P4*tQ`0'
    'rtC}Ouf%!<_X935X%p5EETru7M0F|F4{1wjB7hFVJQCr<KCb)P`Qh=|htt~ogOl3JgY$#h`9Dq$YrnibK8N}%C0Tj+u_e`ETIs-z'
    'rrG!5rNh&s!-J29wc`)Rhx2!-`|8=RZ~ytvqr)0zip*AfRH>buet7@k{OyNhyr?|h&Xsv}^x@#V2HUpw>g~}HHBYl>ubnZu`~C9r'
    'pmz4lix-DyXT|HMUB|zw0W43FHaMMza=YC{`bq{K;Ml^+aNNA8R=FPJae!(&x0J_B_CBYL18ye!;IhUP#f+c(d?wE`w+%)^zCy|W'
    'oUMm%7q!zJ%xdj1x^uYqhto;+)>T@oH=CZjy~5hriNB;h+n<}CN9O07FV=4yTXk__ezKHqHKZPlhj@soE0gm_-NCcgdJg?>4X224'
    'LE{2b)2UdslD1$t!V{z3d+L15e$Z-kOij(yuC<q25X~`$4`*+KK<D{}NxyC?8?|dh_pED+>?<QL+jlWN`f};0SAlW3oNZv<Myy_}'
    '=$5P3o?gSc5ZbOK7s_ZuuViG~)NiLK_Er~{^<9+PdWVuWOQC$PX3M-iD|*|>lxZ&f-zED7g*g6YiqX*Nf-_xkHou_SQUoi<KZ}Z@'
    '%c$oYvoi<9@?Xi@5u>T-M^O{FKe<qEPLU>QXZeSJoSrKS|8sOSd-FwIzWGAev=I5m<!qPrJA8eTrO_uUTm*j6AL|sz!?|pJFbXSD'
    'k~A=senb(CoKxe69L-m&2srhm7gERyc88O9KMlh8=m_y^He*aO%j=3#E>ZpMF8TUy#z6zf#t-tbJ*^@TC)_Z`DhSyJsysmtT)g1Y'
    'MtvopY^^!`k}y^S&8kKz#aB31NW;`!{Jm=@CzJa+ZJLrF<UanU@=aZ`$*t++e|siHE{B<;S8z6MA*!5pWLk(=WxXB^TZQlpHQ!?A'
    'Q>3xZJt42pX?B*f4Wh>5ft|PzCdW<6n0-S{4tj9zAu-;#sHZjZn+>u1xoxC*XV^kee0|?Tttj82;ejbZn2cjH<#Fwqf$a&=Wa5rA'
    'lYbTMtOliJfIt0vs@JN0F&hWzGcA{OwO%xjs6)2&!p;tFXqSE%IG4`;QZ>xoUeL91qi73^8xgsiTLKB~hH*oPTD!ej!5(KfrnMI<'
    'q!FIYD`0FqDt`vyY0w*KRUykjV=1Rkv`2#$)`wTGHVINe2&AVKA_=-|)yk<l84kI{JiU329=|w@1oQxlbF2LGHq@gmC3qQxS_2OS'
    'NhXv489~igSOdDnd^_$(;Obc73vfNn>APDe9W35TNiT1xn72%(WjxCsSvi`s-N0?5I4)crv{NE;56fsjI6Fb9X6Zz25>L^>qMrj#'
    'X%Q|pFNzgt7xPDklVFxhzfx1FPWgU@GAe!yX`hCzDN;#Ck~z57GgQ&z;YV7hY89VUtDR)>Vmcltg9!>UT#_k<<>=P0|1m_`vK}iL'
    'pYJ?1VLabw+@|Ur^Y#lysCrQ!)f-)yk(>8iMAdam!w0;qGR0!_7#{O@i-4IBwUK%Wr_j%i4R@sU^2*qvGS-IlHi2w$9r}f<vH>cR'
    'GDuZ$`#sjkb6ku9JLC<W%!;1Ei>_(O)xQu|Sj=7U2tqBmSd=y=&(xMgmy>^QX1zCC?mb7Z*iZUE{lbUX1<KGT?w5G2{A8^%vrL&;'
    'q$kcQHJW6KjV4i)v1g-4DF)=L$=QxPGrX(v=vp?5Rx(Nk<P0Ewk0DTS#I$I!PDq(Vj>PQ8rJly*e#`n(<rFK2bjfP5L`D?sD-Dvo'
    'ymPmgZ&&^=51|O%'
)
ATT7_CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes);
"""
_ATT7 = None


def _att7():
    global _ATT7
    if _ATT7 is None:
        import os
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
        _ATT7 = load_inline(name=f"muse_att_v298w1pf_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,
                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],
                            verbose=False)
    return _ATT7


_WS7 = {}
_PF7 = None   # (pooler weight stream pointer, bytes) for the next attention launch
_NUM_SMS7 = {}


def _workspace7(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets
    them), each followed by sentinel words -- see _guarded / _check_guards."""
    key = str(device)
    ws = _WS7.get(key)
    if ws is None or ws[0].numel() < n_rows * 64:
        n = max(n_rows, 1 << 15)
        ws = tuple(_guarded(t) for t in (
            torch.zeros(n * 64, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device)))
        _WS7[key] = ws
    return ws


def _num_sms7(device):
    key = str(device)
    if key not in _NUM_SMS7:
        _NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS7[key]


# per-shape launch overrides measured on the judge's H200: (batch, q_len, sliding, kv pairs) -> plan fields
_TUNED7 = {
    # (Oct 26 14:16-14:21, podbench_multi 15 rounds x2: b32 q8 sliding at 4 splits per request -- 128 CTAs, one per SM --
    # 16.10 -> 15.50 us; 3 splits +13%, 5 +10.6%, 6 +3.1%: since the faster per-request q8 loop (LATESYNCN, QKFIRST))
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    # (Oct 26 14:44-14:48: b1 q1 full with the cooperative finalize over 8 copies: GPU-1 plan sweep -1.6%, podbench_multi
    # 12 rounds 12.30 -> 12.00 us; coop with 1/2/4 copies +26/+16/+4%: same-address red serialization of 250 splits)
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   # (experiment: 252 instead of 264 splits)
}


OKC_FLATCOOP = 1
OKC_PDL = 0      # launch with programmatic dependent launch (for a fused layer: overlaps the qkv GEMM's tail)
OKC_FLAT1B = 0   # q_len 1, one full-layer request: flat kernel (Oct 26 00:14: the per-request coop plan, 264 splits, 4 copies, is now faster: f1 13.2 -> 12.6 us)
OKC_PLAIN8 = 1   # q_len 8 sliding, >= 16 requests: plain split-KV, ~1 CTA per SM
OKC_F2 = 0   # must match the CUDA macro OKC_F2 (q_len 8 flat: 1 CTA per SM, two streams)
OKC_S2 = 0   # must match the CUDA macro OKC_S2 (q_len 8 per-request kernels: 2 streams, 1 CTA per SM)
OKC_W0_L1 = 1024   # flat: first CTA's share on each SM, /1024 (uneven shares measured slower: the faster CTA is not always the first)
OKC_W0_PR = 1024   # per-request coop grids, one request, more splits than SMs: first-wave tile weight /1024
OKC_W0_L8 = 1135   # q_len 8 flat: the lower-blockIdx CTA of each SM takes 1135/2048 (Oct 25 re-sweeps: bottom 1130-1140; 1180 overloads it)


def _plan7(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies."""
    sms = _num_sms7(device)
    target = 2 * sms
    if not window and 1 < B <= 512:
        # full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair;
        # several workspace copies spread the end-of-run atomics (H200: b32 L1 206 us with 8 vs 237 with 1)
        ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
        # one workspace copy since the 16-byte merge atomics and the parity barriers (v101: q_len 1 b16 91.4 -> 90.0,
        # b32 167.7 -> 166.1 with 264 CTAs; q_len 8 b32 199.6 -> 198.0); single requests keep their copies (contention)
        return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if L == 8 and OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
        target = sms   # two tile streams per CTA, one CTA per SM (the kernel's OKC_S2)
    if not window:
        if L == 1 and B == 1 and OKC_FLAT1B:
            # one long request, q_len 1: the flat kernel (producer warps) over 2 CTAs per SM (H200: 14.4 -> 13.9 us)
            return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
        # one long request: plain split-KV over 2 CTAs per SM (clusters measured slower here); q_len 8: 252 CTAs
        # and 8 workspace copies (H200 b1: 20.7 -> 19.75 us vs 264 and 4)
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        # (q_len 1: the last CTA finalizes, over 4 workspace copies -- Oct 26 00:26, f1 per-request: coop 13.0-13.1 us,
        # last-CTA 12.6-12.9, one copy 15.2, two 14.0, eight 12.9-13.2; the flat kernel 13.6)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:
        # judge timings: plain split-KV with one tile per CTA and a single workspace copy beats clusters here
        # (Oct 25 21:36, after the cheaper merges: b1 33 splits (one tile each) 6.6 -> 6.5 us vs 32, b16 16 splits
        # 9.4 -> 9.2 us vs 12)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16 and OKC_PLAIN8:
        # q_len 8, few requests: plain split-KV with one tile per CTA and the cooperative finalize (every CTA
        # normalizes a slice of the 128 rows) beats clusters since the inlined finalizes (H200 b1: 13.0 -> 10.2 us)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)   # (one copy since v101: b1 9.8 -> 9.6 us)
    if L == 8 and reqs >= 16 and OKC_PLAIN8:
        # q_len 8, many requests: plain split-KV with about one CTA per SM (H200 since the inlined last-CTA
        # finalize: b16 19.0 -> 17.2 us vs clusters of 8, b32 22.0 -> 21.1 us vs two CTAs per SM); the cooperative
        # finalize helps at 8 splits per request (b16 16.2 -> 15.85 us), and at 4 (b32) since the parity barriers
        # (v102: 20.15 -> 19.55 us); b32: two CTAs per SM (8 splits) since v126 (19.4 -> 18.25 us)
        n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        # many requests: clusters no longer fit one CTA per SM (GPC sizes: 32 clusters of 4 put 16 CTAs two to
        # an SM, which then finish last); plain split-KV spreads evenly (H200: b32 q8 27.7 -> 23.5, q1 14.4 -> 13.1)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    # sliding window: one cluster per (request, KV pair); keep the grid small enough that all clusters
    # are resident at once
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   # one CTA per SM: clusters of 8 fit 15 per 132 SMs (GPC sizes), so stay below
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and ncl > 1:
        ncl -= 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))


_CL7 = {}


def _cluster_bufs7(n_ctas, L, device):
    """sm_80 emulation buffers for the cluster path (unused on sm_90 beyond a
    pointer), each followed by sentinel words -- see _guarded."""
    key = str(device)
    need = n_ctas * (L * 16 * 64 + 1024)
    cur = _CL7.get(key)
    if cur is None or cur[0].numel() < need or cur[1].numel() < n_ctas:
        cur = tuple(_guarded(t) for t in (
            torch.zeros(max(need, 1 << 20), dtype=torch.float32, device=device),
            torch.zeros(max(n_ctas, 1 << 12), dtype=torch.int32, device=device)))
        _CL7[key] = cur
    return cur


# Routed-experts kernel for T > 32: copied from our muse_moe_w4a16 entry (same author), own extension.
# achitaan CH: lossless CUDA storage, bytes=153989 sha256=230f1455dfec67b6658c8c1bb1d76ac03c80fbcb828878aa0695f9ab8b8d5159
MOE_CUDA_SRC = _ch_source(
    'c-ri}d3)PNvM>C<KShtvV+XJ(njmRf5iRGHY|HWS5+6&x&OEvrUM`?Si2^Bt6sb5qpZ%?0RX5NLfD+{;lT4f$i$J6IuCA`?+HT#y'
    'ug+$1*g6}9t$Dp2G}J|PQ;nifjl)Sih^JvRRj1)33Wu9&GMY^X(UBTYM#It3EG((#&tAU{*7oLu=da$&TXiiOP0mz2ZBOHp>JOs!'
    'P$A`hJUc7j`maIM9nN~8deoiu+O1AMXq1m1TglVQaoCzp+JkBQIGcMii>8CK&@B*8dxL1oX}yq|3)L?#pR}Gn+j;+7Rh+c<d(T??'
    'ueVj;q`rIg%l1<hC`B`&$G554!=ajv#;2`8FIK}*oBj^scoz0lC+v?Vp_(3tDn1J_PI`@<8h3+d+wWfeQ`boqUaHP)Fzm&fYIl2Y'
    'KTz}bWGwYjwbPhe(yi#~hi$5>@&moMCzC-qQG<vwI<0(ozPJ10&E6h{Gw4V3g_B=XqLU8^=nDEmWj3*{>*0d_PE~hwHXer4!6>3f'
    'np|3OYKhjo-BY7J5AtX-=q0_d1}_UcZpYNCFzOBZPA_)qubfWMyQ(Ow69f$FqyBn)K5NtTQ9YCCpgoLLe=<7b`p3uZI9%UR;dr1y'
    'EvA(jhU*h_?&{WG@4n4U!CMtPD9*W;$+?&GGipbpco>Fb6=7D=QLm$2g?>;mdK!-iTD;08fq4}A#m)RrrkVdy^@q|-pn6nno9AFK'
    'y)3D!TA=RBB%w|fJL^zV8w4S<sC%sQsQc=-a5AC^DydpkU5tk9>0lVDz4B2Q31nH_QvcBKbL+om*T7B1&QiFj#jBsKj8ibPG~iA+'
    't+%?=|6aIBU^kqdMKq%m+H&F0T3FzYMc7*(O?m`*u>~XEXdcOqQthcCjP*#pR*$x5<hjY;skeA9w*HxPCr!f6;p{0`x{$31+RtJY'
    'wrQe)8baET1i08525`M1SS9p1yi{{q8Om|#J99(x@Z=YF9wjO1O=g$BriuH}N&EHfyJr>>W1Q03e)r_(3@YFcZcB`QYT1UQhF{(&'
    'FU|CrR_pFvF17#S%|G#MHKf02jOvc?$=lssr&T0aqqBSX<d?nuogKH2l4d;@%?p3ovvRrE|DSI%b3rLDvsqrMdQsK;D4m`hdoS_s'
    'nQCd|-anFA`Th5#h=1i2nb*CS`)X^8_AkBvLl$#4d4BT?&LZC(=zeQoHR{e__TIjF;SQKn^!|!Z2T^pn&M+vuWImW4D?#}&fcG&C'
    'ifWyXPD4U%@pw3x5`vqKu!-<xoxiA)QHKEh^^2z^^-FiEs*OV2hIw%34-->}7v6kuGQRnsHj1+G_H^-wBo%Lwofpqv^9(s#WEWj<'
    'p)w1U#ntU@@9*tz@4tAGp%WO&RJ~FFzy<Kj8S_K?;2t+0_Vk*iM1s&>;etpYbsP?9%^Y$8=w?EAo-h%hsyiAFD778+6cI;+sxHFG'
    'rG|h69Tnr1y<c_Pz3^g_PN6MeKn4}UbllGT=<KYG^GH51mWxsOcpUcJVLji=)}9*}jZ3}TTjxqQ=}ha6BEqG5ct?a@#|9K%0!AD4'
    '^~q?iW}tgYYCN3LD0#?4d-i5sT`oJcy%(?Ex1PTEKOXhF<EvJy-y2*E=%eSg{gU!JzKg5gf7pKK(h8iSNjI5z%DkZYpdKC_Qy=2v'
    'S-(F7Ju*_`>7pI0ZaX3hndk)MU#HV*a6yFPMSIBfU4${<AR&t32=w&CIg9hXz)S1J(_g(Q1`r?jhpqTB>Uxz$U;j6wk<iD-kJW>c'
    's$f?R<FK&2XYv(p-M8;v@9#ZvhA&CQ+n3I%%qHR#CYTbulaRV{jM5Q!_JdHktvQh?<56r=kqJRVhwYlW?9BRx4sK7o$25zcU>;gc'
    ';uQQsNZxz%j=%{$&=kTt&S<Kt4|0~Mjd>Uf^#*d6PH|o**)&MCcA1Nw7#q`6+=|*~VLWbk>0m$Ys*ksvS$S3c*(pZv&S95iq7J^Q'
    'f><1Mv1h#}@AqGA6aD^#$cyl2!oPcGKUVxrH4g_z$AqZkDUpF^3FM6;YTpYl1~ei4!6csgO(1RRIR?U0M-&MGnNY|?0W#ZwW|>)`'
    'F%YDJQk6P}ZK|TI&~VHIi%>sjZ8|H=)DVL+B78!)sNWxSe*n=N&xVvAErI@QM@M0=oQ$7_+1h)*yZho-b-Vnky1CiK*jmTb#%(9l'
    'FK?f{+kW#@-7cv+b@$WuyXUH;ZVzu0zz5aciv8bgIP`C|T0M*)Y_+hzN0V+ih=x>}-Y-T(B*)Vhb|dqaUaJ*d&=UdGG-%vYgch6>'
    'b>Y)`D%Fn95~|s|9Z$<g!%?R_%#5Mjp%MpoDu>kTfc{mR|2Mj=HaF2(pqJanx4nClkXL0q38$?t&G93`%Y&ncU^zY>O{VuAdv`B<'
    'KN(>?KSH&sZp+Ny_M5jptvuj<t?Kc(c%1xn)A14WaMk$wx4nxSb=}2GHh3(7ha4<?{pD8o+H|5_G}V{StW}lX@V9WRO^MdR-kYkQ'
    'pVIew<wrtbFU(^zlZVKlElm%V)EgoXk#di?nOw&GSNOP3n9?(+>`!!1(xhu*;wNx_n{*^J86U1W6omI-&sV=E=mmRx;hv_1l&DUh'
    '%L;!P5<Nl<8pKJsC<J!)+?szE`oOM=+D^Ek<be0H1A4ckht7RZu=%E{Hsws|PDXLO&J`9Q_XNFCaZQPX5}|TW_P3KuixJU8jLixL'
    '!bGx{IRzAs+gPY(`jn#JGhL-jo~gf@671A$Yww?Lo_Ky{$5GxgYE{@)6|=3P+uoafXV0XE_+IP3e|hm{c}3M+MXxe7Ia|-_S&PXD'
    'Jp*2Hre>CgIG8QK)_sCTECw$J5oIq`_jnea#ww*uBTz+h_=8+i9-y2H5-Df0xm*p{pp8tE%#&=H&l&*E+6%*{5K)-4L#&DNHbAyU'
    '>zJ6P3K|48`Yl!$gSK4HEwV88?%C7UhqM{ug*FKEHfZYV&0Dy%9CTa6`V*@pY+edlhIJ$Sr$10WqF`flGsRafX$NDVK(}#b=C%^P'
    'Zpj$?gk5-m!x0^?6XC;f;#!=VEJAK@JFRW1ze?zX(9M0CloHag%YK%3-DJ>U>bf%=bx-3=CGVvYu1l(TFI8|&TuDK*tyjN1r#wGO'
    'gZgbFJEdNr9z9BO_?S<;(3EoJXR~-p6q~9(XyhkvXZO`MD&;C_6+v;ccsQDF(xjO?7=}PPeEXpgkZs{*2*xqFoJ(y3Mf5I-tQ&OI'
    'v^^LK=>v+0=(EvKxDHtiF7ZQb)^$Fbobp<9hwZ_c)K8q3!o75INgz;9Tkc9?1!JO4w9ul5muRoViO+Ge@Yp2}jrlpPsHwb3RM>Pj'
    'iS*6HPs;15M^u=edIZMN9`HZ~XVQ|n-?ry|TjqW{nfnJp6?5;F<>P5@7Z(H%%~i&mJ-iYJa@$}+tOcN6V2l>g0y5<Rta)OyQ2YAz'
    '_OCR?+D5&kp2;&O-NRn{Z~d`KCklVOFS}?Bz~$ES>s#8R0_ynca3u?cu8ao+6vPEGFX0w`xrdGOpdq+U<2&l|rp93lNHmUUjq6MH'
    '9<{!U%}o`htz*pBHvMJ~<!WimX-RI=-k?Hz?0@*B0k*teZUjMNqfrSq>eYv}jg2ta=r?cC;!S2m;=gTA+GjBnfC5f4YOa1y&9|Fw'
    'LYyRMD(8q&a;0{%rFL?qcHC06YKxMGv#H6Lq0b-}n#)Ee#&FugJF$txQy&oz9qH*J<W5sMR|C0_rA<di&hA?RiQ{k*rVSc8o7j>G'
    '2H9^e`Lfe*nmN|j%R&^H>lClnP%n3Q*-zJb3H;@)%$Bw?(<q8yex0&R2S>BfEG}ro#>Q8)6fJEjYHEoE;RPa*P8oVkJAC6p<kR%{'
    'in`;ab;nI}J+5oqR0n2jV`zs5JVS>E37L0z@FR`vkQ&P@_kzGIo3zt(BvUx-O8g+(Z9)58xLjIu8OaVI>Py^FnTDf{!-G@$$N#5R'
    'DgQk!L$UT$?!R2CG$Ka&i*3HUpN5gCOe8d`R_4hV^noPsW8Sw@YB(W~W3Hd*8QRg@Yu~hc)*oKgsI<$1n*OkT#1~8>+BMy0G;85v'
    'Ff~#=det(e`+6)THJy({dPGZ#2F-qDNs=r|dpK`j#;T&c(YSq1lspk0<{pJ;ok<E$*;Q?(=hwYNkeb03awpW3v8vZ4H5`>xM0=oV'
    'hTok|n$|#Rb?IP3Djiw@es^#?ncN}Oc+Ae4HPga4ND5$*flG#afb#{68nWq5;$9^oe|3hR?F?}O7AC_X_Q@r6R7+4HZ<Pymq84LE'
    'c*GflO{ydyEY~(dy&3LV`5VM{cU53F7c?`?<;O&u19O;cWzdoQhi0X7)U;~@SJljl<rItSL{jXr;Hv~2C9*mNGS^=!zy$}%Y?XiO'
    'Q+g8ZffoO0Lm3?kKqH4uS0bsqMxfj~AolR^;LV4_gIe|Qfa&%_eO6Gvn9j?yhdjrc_p$2=4)y<&dh{GJ^QVaf;7DAtgHRzdks1J{'
    '7mb9(BLsx8$^E9Ez2W<?r$^VQ)9B`LNYGlM-$~y*AVniPLKL{fXM%~kHw+0i5JP#%dI#<;<^6PO%?=jsSM?`hs5cMQeo#MbUX7ku'
    'yh&#$(WOMsYD$<nS$Wp9^8|N#xhrZ}n`C^n=i-BE{m??*m2$b3TrTuy!(oZYlk&qPDLflPn7vUh2M-9XOOBvXE>~;x%3sZBjzEd0'
    '14@On)cX{DYUKsc;F>c+|4kj;G4FUH90I$S1`Bsbv9Jk)g*gDyk~?-$x8%+hB~R!8DJsdgwX}@A9`zN+CZPb9DTB!)s9J!}TI?u~'
    '4<Mb7Q0<1P-$Bpw>1djn7&9pP(jIaLvjujV1qF8O0%ps&OB_`7<k9NZ%{F1s^)&fUC))#B(tt@0^F<g1zZi&QiCutomiR^hD)q(T'
    'c}ey8FD;Fz>R69E=9KAoqbblyG98KV)ad3|+A(1?om|QhY6ShdLh2sYZvC;AsuBx7#v<X;kWN3^psg{TUAj@U*{Bl|81?&jXeN+m'
    'u$xIac_znHNuBH^PfyHKpJ`QjGbN&vT{<*Rh;zi;m{dA8Ty7$($wLF6JmI9xx<BoyIeCa!q0xA2JP$h2CwLivgX6`KG%&<+L^+2^'
    '({8aiGrGr2yI{kFLRQo{Ci^THWAx~)2clWi)HeXXtBcL0>^^NL4LF*J0*l?%b0&v}eBQJ<eFJv5!|aB{^7cw*@6jPm=f_|wTat;2'
    ';yVS?9%^hw;bK}!AFEgxi>_ATpeKhne}G&7(590#<4TE?9YDxrJd+UjHr6mHKF$gOy`(%bP)>P)UINi^doaP*0oJ;t)&oV`5r-O)'
    'TJ0g9(%^-LqtV1`6Vu%#Z1mhuB*E>7X(U`ki7Zul*drQXx+6+p(KBmqwV}|{V;wS_`>a?7@d6!y;MI8Aq!zRp&pM)U$K(#-$o~xp'
    'Hhf&?pc{`=^<ULG(dEz8{}R1Ny(*!C2we&nZ6YhyXfT9_*JtDFE?UlJLu$8m#1}1Xxo7F>>lSmPKY8w3yfQwGi6ZB|n2mK#2~<6@'
    'SLwYq`OOo-fIp^eQ~_t~EAO0MpDUVtU#+j7BNh0SKhZZjAAHV0Xp(ZQt#J;0t#f{j%qKzY9U4^COI~SI=_eV9Ll;)ngnNI?#qx40'
    'H#6i^kol~VujBhx{|h8w<|=^lPFOAk=7LUruTuumkH0=6dYyPYK;rShmy0=ql6DU@5egha6*JO|$j`6ezpAPc!6OQL<)169pDP3?'
    '<)4Gr&w;=CEDpc*G`3p9(Y%@@dn-=sP3Lvx0Uv++0Us0H_w7e~tqkw%Ow^2x0IzEew_UMc0{f+Eztrq0s+4Z2RsTy%1qFgwUC(Xm'
    'A`5lO?IE8L<+Gp>ZJZKuOcY_6FlxD9tCm?;n*Y$7`go^Ox)YS{R7-bipGx#vFWq@iy3;7#dC0FDr8__JgOv^9BHiYuy1mxF?R5_-'
    'hY;w>b6`DJt>@aIpB9_k_SzME!$V*ms^+1lA37!vjeI33c6Z!rV-jekgp7jxrlgUvq#SWd2j}($O;yi#S<0BeGy!YB)#_eM+bt$@'
    '2Hn<l)Qaik4SOEy^$kLge}e!Qm8qYgN{XBFoV>g{E_2t)-J`NpqUGmWSblIPI4rTGBo&#{$YU)kN4NE4#J{aP8N}Y44W=Gs0n2*B'
    'Ia<sO>1KUg`%j6Pg8GMg#J9_jS7~kDzPc@ZmzFYON8i6jke<JVP)CrKOgz1O`R~*G5}b`L%CknD8mCE?$>&O%w*iw|G|N*5bxc|N'
    'Kua8-?mYKSeInjyd^mwq1A)7DwJ7*SZ&6-AB*I#+<8rTa<jc<iU(3q(d&HBiDXQy`pvH`X#!u7@WPe`8zeUtuNbR>YwDE-Swr;h&'
    'py9jss|@3<Nqc@kgy0o(m{8cQ?u=VpN?|Kc#!N?P9~Xao@iL3UNqns=YS?SFyXUjPB=oGyyRobiGF;@BW!Nk0GUcQ3%xINBjRXsY'
    'eIc+yOEiJ*2%c!I#?D$@FdnB7Jq^!i?PxmqEo3FgT6Mic)cu^;9$bz-tG*7H-i-Z}0;scM*rKYo*ZY$;?X+l`>e(qzE8&IJvw#+_'
    '`mgn%+VIP0>+K{Qx4Wmfb{R~~jj|Vqp=u$$B@Jz=NrjgBT;0V5<pf%ilhJhIotJ1SC9RsK1r<p|{gWE75q%iahdR<Li;ZBTf`3rM'
    '@N^mdsZyy|@Q?7sYbuDmPb&CVzIONa&g<!6ex9CGcd75rTj0D^owu4r`m&%YsnPAujMpuh6MhSXD4s<T(YVKhK9R*VFmy#Gi8ED~'
    'X412UT!4Z9dFZT2g3Rz?OYKi)A+5?owf*WnZN;j3@p}7ddGF^}%&%>j+orO5r-6X@s2NR}zM*7}gmqg9s5YfV8chdGU^^CLo}ON4'
    'hPy|PR6SkfA$7JSOKZM1l-J<$SX9YUm&*cNu!n(sGt<O;LC~08H7~>bya+IH;YNL?fvG|(LjPtAO-rgU&=nAH(Z89s?eu)}&`-nh'
    '6f9*5qAG$av6@fXW7c$K`FPl$Vu)Cc0n~A6VTj-6CO0vCZLnF>!FYIyyHn;0^>pt-ad$8drA^N)nxC~c!m1*l=sAakg!G4A$ed=('
    '60T_&yOW}4H0^%Tq!@V1?6{ebk=V2AU-)0nJ`TE;P{@9=zfHVmkC4-VSZ=LF=^$j@$zgjkyln9elSaWJ!KDMK#F+f9+Vh*aVthlt'
    'yg-JhoerQ-V4Zp;u{PV{FLzaq&;;M8Z&{keL~AvjTAtD8arre)ME3*h4SC)(+MHUAO|665rB&DOzOS|0>|z6;6mIq&RY8-5-EuaF'
    'yk{kRTcgB;w?;o7nc97|SDtqdkOn>H!vix49MP3#a`z}^ep+<__xS1M{~2!G<yxQU58~2;_39l_{~Ao01076gzySHFI>S9h^+5w$'
    '`lHsRfk=|ImOV_%2JKm`Pr@p66FTZcxjzdPlL9!Ud{OitO+*mKDr#7g63YMzj19%k5Jb4EJ@D!+v<Ez`8&Su+@0%%qv)|+rkk_m5'
    ';(+oVHq$y}CF@lD877aG<iEx^MB7BA5Cji7C-p$_t;s1*DE&M_n@zR0b}F#t);FkYP+Fs`_mVCsw10|4p$w1E>Q7*bdq-Z5$xG_0'
    '6Pq_K8}Xq}H4f?^A<o*TVGG|q!p8dlzBKdcn+^fcr@Nn)vrAqa)G@@5JXRtY*J+dBhC|*jGw;7;-sk%LQ!}lasKmP5Kj{RJsYIxx'
    'M5OO|7sSS?$C`NMmq%0)WT`$)Gbe1S^}x5Qxh$$Cu&W8O)dU6A{8m&=)vl(Rt)^N~&Ah0Znq5sTTTLxjP5!=srw^9}du3d?(%Riq'
    '^+TT;OQ!VGE%P@~S9TysZQQaTXdsZ$qz8fm{H#GG*FXcZgnXg{vQyXv==yqfrD5Hn!~D3zWYD2A;xHZVr&RipDt0#Qes*vtO?!=?'
    'ogW_&M%9mNY$>TbYymxGm+yQYWYGj_^9EcGI)g_1WDMw-T_9m{#v#aX%*PoFI$G`MvcQ-n)N6?Cm5yFcC8m!0vBx#}g4h`2<FGh9'
    '2oEuyR%<*OcXb|_pzO?`ZaLPddV9H82$P^mh{W{4aRnZk3m%#CF$6R+XO+E!IWH%7++I29@8JPH9Ex%tnxT#(xv4>r7a{5n0b!%*'
    ')g8bQPV0$tkWNzkKH9>}d=xCiGtQPnvvI+Eqm`2C!GNT%yGxVa^QHMUt<i4!H0t*ON%*ubJrnqaSW8B<WhX<(Ws>PRo!L><q}+UJ'
    '-5I26XwXz&W(MgnhOX=;%OXAMpSE-?8J%S!CZRm&i%3>W6STz1t5B0Dqm3YGcC>I$tB4UjK-4Y-w+T@phL~tNuPzBQNUbWY1&p>~'
    'rU%2g^>}&-T?(c>Zu{9S7xhmB6f?;&l0AkXQJU7LJ)Vu;d^i|#j{9oE7X@M(`y&u1qG{YrgywpXlW<)Oj>IK_=A#m^aK2xtT^o}A'
    'WO~DznYuUpym@8!wT&?z&7|WNPD&*hPD^fqpQvT{0l-sy@|iot9Oc+2!8IGDKfpFAsf+408Dq|9lL^+Y`iRbBS5bdrd8#(8bH|<A'
    'eMUe_4gb)k8m?WV_?n!t<V-gpFB^4bUC-kQ&*(m{T+GRsRA&~?WjCHf!MJ~d?RcI};a>UV*#T9$rp~b^`P4w-o|OpXL63Coio!d6'
    'eqwzFg^v#<27>h@V@l2Cc3NTI;-qiVtGK9FaY3(S&v36|r&oeE(q3tl@X6W{SY~S3YBblNL4u|QgDdS5+=3T<Hc{^AWY?@tlVA53'
    '==Y^83W*s9jx-!_wh~p!+9di)>X_K`XvB!|d=}0^|CXg^%qfEAHmU!UBQV}#N+IeGT7|_UqM0C%a9xMKbS||5OD|j`Fb4b11ZSmm'
    'cL-{eF~WW5M&tc4@6YvhdQL$qDbcUU>h>!PdOU7=c4!xO=irb+4vTt8GqB9y3^`17Y{>5kI1&w^9<rTmjn(3*OW6t%iUzm-*mU$e'
    '98C+h-I+y@VqwofuV~c`%+8vjR!|fA5X^9ogkCdwGxPa&xu4d;rM9eMewKDkU@1@c*mnxJ6!?!K6N{KPpoI*9=+K8t=De98H7H`7'
    '3tbMy3V)D(f{w+&`gE?*R4UvzL;sjBW5!;zeZMaCo3A|!7r5~HDmPxK4zq{r3|o`EWQ>}s(^7T17Jvc~q~h}h`9Y;zCXSMo-j0-a'
    'glon7&|PI6xd>O`lY?rxTr+vrHk3CT+j(#f>PDugz!=n&;<hFsL=+dIvP?~0D_y~zT_<dvdyd_#EupyX9-gM6wVl6)W30$}Rq`BB'
    'hfA8(IIU<+l+n;Ap&Z5ClB*{@L0595PO^oM1yLSDT?U8XvCZ5}Px-((1ztQ(!EA~f5Ol=CywW--ipjz8p}HqMnT<IC);A`x>68Gk'
    'k$7D=qA>>LKJPf9EBoI1(Uh<IW@A&zG1ncJMYn&Awq!i`Eo@DFPv(k}rl-Exv8`c#SfxM|Z;Z--XO`^3_x+~D#<PBjEM07=Ut21|'
    'juv)R`)crcQngquf|4Ju<^;Z0TK%2RTm7hV#p183Y+yu$=k|TC)H=t`Fw2tYTktQf0NdakM63re4_3(u`=&#``4wSOgHpu-KKeVS'
    'nPDQ`g+P4K$TU7j{w<Wfhd*n`e-EVgRlsiBb&t#JJK+itWP38kM;8^p`6X3g=1@(L0oQBo(Bsx5gmHDN-Rs>P$P(C)#G}tdC^FY;'
    '3Cpn<O^VnK<bsr?dGkUWpHm1HV)}@~avvo(JXf)Y`9WH|n*q)QZran)*`T{kbAQ*!;1ALj*0~RLk92Q7eVqeVTadV(7{AitZJ9l6'
    'D$!S1*DgP_K^)X@(bTpm+HSFJ6PWd&p%G)rwO-Sj{pzLIY`)wPi9O9v9WQaWJq%;UamE~O5Gk!pBTch({=G54#qdQRoYlUwPEoWS'
    's(50A^{MjUBd9YQo*MhNQAD3xKhicZ$}CK+u|BG(0&L@wSDRa~eXkZ;vda{VyCmxRl+3-FM=+9Ie<@umiKJ+u505td7fD}_Lso@N'
    '^%2Uh7q2G`*J~0IGR|~FJ-OSS(2_J*8$xDKd9}*Js@k#Tr)+h<Q|Rc2YW|RI>Vs(tOrqGC7B=*-vEjB8RH}8glNk1kQV>mbH(cUN'
    ';QgBy`>iLxJbkvuKtUCjByI1BF~rBne^NZAP3<p(bL{+cpG#R)teLXSTenuycVr!#8KG%gl?-YvhPqopXKXC2|G@E^1SRYGSxsh&'
    '*kPhyJD8yV2i<`G2i0rQP4OfH^iLXfT(S~7N)U709$tjn%>-y?)XGY*c!7j`Q+Zq|bP#+sp0T2Om^4fCOF5y=Ol8iXHxtM0R5J!`'
    '6W>lSNc(L~1sio#^pFPK8L%TSR$<q5l+`m<s0Iv+s`y37*15DT(T5nVRu+|wa=Eg2_>2v_n$Wu^nkw{sBjMEU$I|&g*Jh@%SMT?Y'
    'D=>PU%m;Cpx(?cV^3=n;vEMF_s1GXMA{aaKIGIk(3ZeYUq7IQgg!^6x2P^9~fPWJ$02*jz@!(0Nf`2pSGq8(Gi8BwQp}>wo5~D`('
    '<K>;wj4^6kQ{ytsAfbvL@mgdR8SD+wR<E9I>kN!fv=Rf9-UnT2cK2;Puft%C{iPXvNu5GX?DJ;8<Fc7KNUqR;Z>W3pD{9s0Ize}g'
    'a2?G-od|jlG(9ihIyYBRTEk@%(lf5UQK?iI0}v4Rao~E$2<l4~BI$$iABrI^8*lydhg<iP?%>}n#eNyTa8F)R1s1BQ&j`^#ZNwK{'
    'bWQM6&+S>UU5@Yp$TNiQX@elGrKArkm_D#=Jk1ofgRLf71?)wJ4J2w(`Z~l_r*5>MQ{WX^W#gz5S`HXkp5!v)qr2Qo?7cg*z-(Nz'
    '=3-W@fDu$T^z{xFJa5u39L2pQf_qCxyJL8YRq_H!(b3`;a6I$s>UC5rbs-~V9@z?O!a{TEH57k#pUD4aO-jeTB=s<t4zV`1>ZWr|'
    'EUEUWF2bf&txEJn{wG>T-aODK8{cXb`Jee#Wu-Csri(dhS?I92VqU5SZ@Xc5>|L>vM5m-H$%L}r7}xV0aCX#!+iB031m1)7n7Oc+'
    'MbjeJV#Y;M86~x{^yX%F+Aen*^~ATBtt4$uXDaAxEorHJB}HlOC7Qel1Gl4B_D)LV`qwIP7ngSLrk>fjcdS~SnZsl!iT~y+Yy&?2'
    '-$<2-f>dQ%{+@!OxB9S6D}85clOB^1O(&P!Qdw%u4Q*VqXJ7q){}`*zq;0CAq)(P{z|xg5-ObE{rE<<U?QeIkS2_o4u}s%Z60`!F'
    'K-@*C7zq}vd3qN->EKrR?G7P)_Aj)_9&cRDGm*5g#Krx<;xYP2hnl*Ns!H?+R}FM5dO8WG5dEYDh5sjJ=u{Lt_%?v0Lnnulm+G5z'
    'n#(YAJn5n!cX?c%@8FoC-|ZO=I@EeEsWl~#wfh2_-}>F=w+^q}2oM5yr>*H?*_K#g*QWi+z{gT^3b1KugTon^B%?!adPpC%oj3Jv'
    'bCb{I+m0te#|Yz26;HLp+pb$d%km}pFEbLhtJD@7v<mq_X$uY)AsCO9Uv)#Y+QZ@3-V}VVkui1ftu`M~7g<15LakB_3V-fsK)kdT'
    'Ey7&!N<%qGsThm0S>4h~nFHUuyYLU-0qxC`pAWML;YbT;S>3Uursnmos=1pViXdAewLQVbj(*#tH)PC8#b6<KlwLVbM3p2fp31oC'
    'n2hGO#=zShpQRd5tWB=P&3$5w6MiBW;)y~fgifZ!!p<#1EGcuS<(jyJY&zXBZk!n9Ttj1^nS`qF8g-dpASEK1@2RO@sEuJ=xOL^i'
    '4OYEW5La$pTaLTnCF>p<bPWZ%?h!U3uK%HC45T7zb#`JiTLb%ESYC^m-WJrvA2M=Pnc<JCk4%G|9^K-A$HUsHvt{b4`l>MFODuRx'
    '1%-IqNPV7&b4qq=C?YYratc+I*laDY_q05W{cOPP5c?g&okKrpeEvobT4?u=+Q~V8LP0FRqMdTDkLlE+VQk7Yh))sCgt7A_a|L%m'
    'b9hg9q&0dZzHCE^Ea2Oc7v(0~-cbV;MGpvkH`p#iD}xpZ&4C^U*oTc0gUf&rPNOCuAXw|-8bASp0F17TS<Dt#hD$b5e8MEgGW@ae'
    'jc^E71AEU@cah(-&OY^R;d!xPFxntU;aR%JdniF2TVlNS=~h-W{)wG=iz~%zc85q}_%fh{b91T1g+80Lbnha_9nupH)Z!e{F=xKZ'
    'j!bUdXlFnfLmS0*$NiR({B(DWY<VZrQlY&zT<(aL&S<zp<-WdcNW$32T^jCP%w?M3F@5PqU%k5A%Hg8Iw{N8MJsGaus9v@P_D|t;'
    '^sABf(R)BdK!p5-_9H6#oBF=};G!ZzIx$0<0kxim;xOdE%w0()6hh<SkWh--yZG4d9lJx_Onh&;ouMVibqf7xlPu9Uh2jowDfMME'
    'p}f)JQnkD6pG3<>2C*Cm#_=1Jx!BR4u^eGKTNj07#Me=2EK3?81YHWfA4c7<%*8aRZm|WL4akuAiL+*Lhr$k$hQm!+Vy$@wJ=Bu%'
    'm-0g_4p-UJi{y?~dnit>*qw{*4M-Hw#NSJ-?NO<X9iwb!sd>>Z&|-xrI<^X}1O3SEaOeJ_f*N?QMDi558K@1jkbC9SpAapk^{U4&'
    '`v#8R=8@cZ^PQm>Aa1%KW-B0Dj%bWm*+eKHl1H51{UIcB#%vv4Wkl?~hCaewO@zI>PE_ql?~L*%j&eDny+|~Wc#^^e0Mx)Tw2l~>'
    '6W4p=?sUR*C4Sd5LmG&0aWHf<NEzt7qC5w84?Q2!4s`5ea#~3jA7LMvi!4rBTeH-i7RSj)S2c_=7c}j7T&xuJlRR%)(pdDel+bO~'
    'vf5*#LtvOTd6A425+Ip|>10MHc`}1dt4sR8(ccPMPTVW;aC!ZjEGA@~682fjpc{o@FLg)EnFZ*Rp8W$1Kwh8iB)x|~=N|*NSlVjr'
    '-st3nrR9N1HV^<sR(>WJIHA58K%l=KKTEoGvZe}8l#;Bp^i;1k!{0DZ<Y3YP61G0ug;CR6%4zcUU!w}?piY@><~?Ih_o4qP9;4#n'
    'p|Xq<5~Vxrcv$TNA3UOsoKB<FDi%3um$p6co1K+9kW9AS4daZ;ODjuquC3v<>2eW8mDoCgpPE^$?5Zq)=A>$c<P^a-7>z*@(8NQ&'
    'e!|JZmPdC?0o#xbbM=1t+Hld;7HDmax3*ar!<sVGc*zVJ4GlVd%>aB`D^dFaChR1le3TZqh~j%qY-)``rgj<@CyuKx2JxUX43nGx'
    'MJ2fgNIbF4CMPZ)gof7^9ZuBW=B*;WgT4xlm6EO-LS1UCZ+z1kO~}o}9hvgrx7c!Q>>D11w?75|y$1F=8|lNMeN0D4;<+-KcUcd*'
    '+%MrdkZ1RV3XhK1o<b97zP98vcM5D^!3MKIbtAuNXQ-bR<S`>yTJufiB_Im$F_3dewf`@}d=A@AI_K<p1*+Pp1(nR$c-;e)ww@qp'
    'J)9*9X2xcY)y(*o^xVi2X^ikQqNm`q%CqWKswe~^5YFBV{ow4fO4Hd`iSLWdzOgtL*(`Zd5}xZs(l#88PG{pHjKCU{ddds7?4iA9'
    'FL;W7|D8_@R$;%OX_jU2iRvQ)<kZ?nbiB23tJ+f@1p5N1@U56S%+uru)3AQhqq|#rv73cfDn&<9ayy<|b_2_i1<R2I%X)?*3m=;M'
    'FisrcZiXivAj&VB`5BX&oy@Ei>;bkWjh3p9Yhb>JQh3jgNSV=m0Vqdb0hG<kV3;TBA6tyztWgQNJ29OWWan9-FF0tSzV5(#_CE5X'
    'W6I#zsA!irme-hZS==M8De8kn8bGv>B#x@rsQc*=QKSN|lCReFv2`Y$l=QWdT<I(-j>#1b&v&k?t|%GAelciYeR5`)+;My2_?+@Q'
    'tK|LUjJulvPAU*gj?WTm@e9o&(kp#jt`wfbse=QA5rXoHG3;r}an(imn^^QTpw1Af4Y@~rNmUKNmtg;!pawrU>JRXKykX8P&YkK>'
    'WTvA^1+Np*SC`1D#V(i72s=g0h^&+EcjfbjpIr&)O1QUY)D>J^;OA1Sk<9#N&X&<CY_u&z?jbP33KfnWq98J2!L^k?<1**ZrxHF_'
    'mw&fApCdLVOl$xwyhKRhCs$c~(e4>>g@3jA<CWL#_VDM|ESjZY;A#2?v4M!G@{9EgTphV=gQAf)2Q!7f!3%^ZH|IamB9s?|+3ANR'
    '%$(kk^xq_X?l+gzc*K`YPhWaecL_x`xKu@-ih$v0V_6o=tz>H^3PZ2ozk2RXeC=bCo=}7DeBQ`?-|#IMD;PC!#b=G0i8?RsvQ~Ls'
    'IuT`fSEu%%WYX^+CZp$c8b7Mr@XhD1G3sQG<aLGSdN<6mC+FCa^YU-_Gyk+?5S@b!E}<#-?jcI1FPHYqZ}!Xl8Zm>($wDH#Qc1YX'
    'Cj6F7NW07&JD)g9<U`hGjK!q@erfTUU7FfwtL4KGF1z5U3+}llwWgH?ZoX(P<0&nto6EPeCGIY9&ClTS6Qd;Dn{>M+y~ONiOja}S'
    'O=2}ogcrvS51iXGk30!hbLO<pP_!<F=~h-iF+hHDTi>Yr&Ek538`i^gQc+z3WCqH*6+i<0z2v{Y@!vW94LCjE^nlX?27o`)a%TqB'
    '#$2aM+z#}sUgvf-`HaPG*V4~Rpon1xZc^bL6;4s%3v~s^aI2^m)L~b0Eqoa}S5YQKm+j|`TbUinXLN<{#P~R_95y*ANRtAcR85nr'
    'I;oZ>)e_$l#vThN*WJA`VoE@Y7aVF+O!iEHhL#AZxI9b2dcL7nB+Zk~mUItGx?3gP7d;)>dlEjAi5P^n#ia?R+-s)MBGT)S>|Osy'
    '=Mrcw3FdX>ZBUZ$Yu^ajKD|J~<DR*SczlEiiWoGO&}w&IjS?I3c*gq|liEbNMV7>>nX7>>Oel}m0h!91Rsl)ryf|T3<AhngS`%Py'
    'oR?VMOXhiNbqz}Ac}RkXlj!h(s)C!MbRWcQ`3~bED=m;TF^tRdPb-PE38RJb_Dl=7-J)!%$DP?=UyJuJS&7kJ!L7AraOViAGh=5P'
    'J_r|%;_ZWO2*o&#Xx3bo9es`~j!kMDxYnpE*_Or`yV6NVkByk}E9^^Gv2XSL)z8K3B<vJVf(7$z8QC!m(a}+}u*Qo@-<l};iwieM'
    'GW*tqNpg_ZprQSoud(`UM%iaON`4Od%@r2+8f(b7*qE~KH@~Eg#G0~SF!eAZT)~1cHB`Jok&T-du_$(Jrg)Ti8&0RR$j(rYo2?7|'
    'ywkGeRuwl=$rUNCmn84n8sBr8i_bVb7g;lAFZq($$gP%$GFc4l<qx21&J?*$F80!K(zmV-E9u?g$uLd#STZj+*|xCD!n7<DJLxy^'
    '@Q9v<z_lkQMHq(zK4_U2<oX^JSw~Y^cNdhd7L`F&Bl&ECG$4x&rj`saz3)0fCF`oaly=GW9a6cZ`w+z?BS^1rjj%gAoKr^HM)K#v'
    ')hUxdOZs@q!n8zWxFumi^M3FWh%h&&vdE4EoP|+4c|k*AR=#6*9Z(3T{9Wbd@2GNP2?9k8Iq^vRT`lH6OD$$)>Gi5N@U{fb)tLW0'
    '*>#~N0Xsu`Z*&Q=?c_BHJSpI>${R7}OmOXYarJU>bxtyO0g*IJZvG6?WM00E=vWDKE&*;!SiDrGx{|rkN@CSw>1QfWy-EOT<f%z}'
    '--Vvvg`R&Fq32D7cwbd|msC`{rl$ILIpB9W;CDITcRAp{Qx1qmMSYqu5&((cgH$Xi2~|m`iJY+hT~PR4Q1~|x6ka3HvEqf)b+i!B'
    '52vYs5RXC%n5Q~D*;LkFUT~Nb5iV!DZzd#M!g?pI**thopMN<g?v(d$qTuhS?SB_deiu!C7fpT_O@0?mejCwbUK;uS>xv=^g^m{$'
    'bGc?q9F4&7zJ1$u6Jg`G5H+qQX#C^FcFf{^75}pke&8B;6K%kI`-6tuFY8E6gx429D@aQ{%a&*nh-TM*xxZ#V61TDv77!Ao#Q$xQ'
    'XAr0TOq|^URtPkbZk5%GG>DE8CrdEmrNNx8YiFyYzN$6gq{)*ZiU(b)6b%}Md~7f_cPsXsi;hyy)<rgk0Kk4S2r;6ol*?6fkFRA)'
    '5Q>Ru`$Bx<BMgZ4B*J~e82N#X?-IAU;#!o*Ai{rr7!|O$+o2wsG*>H^>*b6$lQ(<2TJl6s&Cl^cLPK2UDvV_c@-yqm?`gb>TO=*O'
    'joMQqQOSyCqglrYx!#Ms^SvYQEOy<63V*UZctzTSSH!nENpfE&_kEww5_6c~0u2qX(Qp!9nx}m8U*Py_sbeQWP5)M>Uw=}!U*FR4'
    '*H`rWZSYMjaYA|9N4)K|3v75_CA67(%Vrm@>}0g9dY4ffeginNPtz6RVRt8f2H<}HU+F-fz?U%EV#nrj684v|2FM>qw$eJ1+AP?X'
    'Dmx=$vfC5*+x8@MuL!O+;$ew)g7~u1Y*)ajk*2uj{^DtzADt0TM2q)hrR{G5k7I%okK83i=y%~uDpvS9$E-jfR4HYOyRCRm>|h#u'
    'J8t)S6QjWMa{a7*GJ=nQD*R{1^TBVw4GB?gu(Nb_njNbE=)VolJN5RNxQg@%C!j_!gT|TY?(VMm5lu3&lLvVHby7Wc(k%ty1ft5{'
    '0~}UxKU%%cdu4X0nL%0(EQg(u^%i$VPT-Uq4#xASx?GmR0eW$9&RTUP&zv+aO;SQ#%E40%GG)>AV0^_fd&6FVf`?kEA=;Q{gXz+i'
    '<a3KbVuc3N<CD+U$E);!g>TE+$H#)!)t!pL{n2gjjf>v%{n+1s4BRm5PSuC!_91U2KMEc*aA*tZYrUtQwRJQe?{2%>^$$z{vBQJ<'
    'p&h-x?ZmlnJ8|yYPMmvOC?>ECz%D!~lOv<h`)V9c$K@z2cWPA-Ak<pLIcM~=;#yNH30bpRK5NvYjnkk}CSJ0P+W`Kftl}T<RB-Uh'
    'p?0VCsiZ#MsngQ~`q!X;4=Lr&MoG7=oBw3@fL3a)e_LM<9&ltAJ0Zv>RI>@SLqDkuU+DO2SA6pnBu~}kspi|Q(@DycYmUTa@KF5@'
    '4}gcDaT9=`e_3$&2za>jBXV)p03Z@nzcq|l2|>1wAX`U}tt0s3;H>r?&H!iC`j^0&V4Vz<ePl4Z6Cihh?7;(uy~bYw_N;_zwvKAH'
    'j%v1!>K_Nc_3!Wt_^nmHF8pdFnT6hu48C^)ln78H_z^H&t^8eJ+DfQp>!@YxsAcP@{c-!^!S{WEeNnG``+cF2dUjWQ<ZW@M%C)0R'
    'wZ?m;{(Y~kzE`Y-dbW;wwvKwXj`|<B-x}Zd8}GM`f6so?hfsF6edO(Sr;0LFl&La{Sl#%(2mjbTXeB(z*6|=)$AfGg5B|7){P6oe'
    '#y)<~`19;zeQ0KP@kidqcd94@SH5>@%-h$h-}m-kZ*N-(jcgr_Y#ohk9gRN@KWu!*55NzN`kxy==tDk>9X>L8xKl%!8p_o8PNvrQ'
    'jx+wAIKxVKn62YswvLC{Iv#!tyaVAbWX+-hqea6KZ!^)fVH;`Qq!8S05gs_7A<XS>5L|238mL>$Cheg*75i|%K)}1D^<4N9%dlDZ'
    'K*yUlYBXA4tD@1m7S^inXcAKDbTU|&NS_B0VYO5GCvjwG@)~OAP<gL*e$kTN_HYy(K}%_mk~j#Pq`dJ9L}n{#zCaUVHVs>I<NqLW'
    'dT>Q>&|=fF__HJOC@ZLsh3qR?Y;S!FmYoEOW9yoLJ`2$`+EfRp9Yq_bQ8!bI`8>=>(3QuEQK}1IfhawKPyMd?7lDGhs~*D66yOw&'
    'p-)VCboQ=-(m^auo{Fv&QmHV9x*hfVbYLK6c|2VnGOWPs!h(aT8s(5hoWvgGuiHRLJbWFbPr4&<6bX`~rwcXd93J$cz#T?YxFN&o'
    'g?YJzg2i++)4rMsjM-nS1l^v5(*Xgp-5&HbMAKWm!1hOoKPz#WP5r9fhxT!<9@hAfP{^$AA=)2iCyAn$7a<zJEJm#FtipL}bZD#&'
    'yT)*$+byv3$HaBnmMw44Ij`Kn9?=*i!jPxS_J}oOj_6p)ISm9iX+T<lQtyqLfmhth=X%A>kBZg7i=E)2rQow$S~5JRy{oKDuQp^d'
    '0n2qjbjItzr8*|~rlcamwDLMSbnY=?DN6_q#7$ohsuktMvyP5QE<r8_QFl1&g@_dmwZsr1%+ofLNZ&EE7_0yKv8s*$!fCC9X@-N-'
    '&~&6JN}WgQcXhP3p{A4eML0yrXB<=#%Dvc5@#T*&m#QopPd6&~H<P`tqrao&61VRc7as#e#FSDSRNB$7V=+66#qO-KsMISQ51VR$'
    'Mhxp<i<ZxZufq{nc6ae_#;HUAQvZDUyTs6P99-RlKza~-yB@?xm;}4kF~lv1fVK!KL1iW;rpq(cs<rBDdF?^AAko11=hVJDld0WU'
    '!0x_&Udb82R($*WMn1!an<L0&2;3U-88!j~LrW|JOX{57&*}Y~-p>=eAy^4|cjp93{*|VJI$x0FbE`}8Z6CyRVrj!XxZ2#(b5f!6'
    '*ZmmikC>2PNd?ukf@-dUnpHt@57-<q=|$k3XL}(TlAchG;EIZ>X+_n-ifUFxY=JK(h8TeLykWlt_J_c-kw`Vsz68gn052>v%%fa2'
    '#YKnlI<R^0BhIaO$|qPd55)I_B;DKuKM&Hblw0LeZfQ%og)HS(uSDHh+DcF=T**@5@|6k~t5mo`R;$TG8%?~_Y=kD{@fNHv%tUH!'
    'xktfc&kF8D$LYlY%C}6u{g5G?#D~72l$dloYhNNJ%_M|lsc9IkZfLrp4NaMn`_~qaji_51(<y`azuc4{mr~88)DD}Dm53Q*<#8d<'
    't=222Lr!0uZrD8lP|59r7$yN+TkTkE+dQuNuJ8sj^X(qdu4|QYrLVQBui+RsRn<16fQB>^HVg1tVwu5=V2sFPL@Nka$kFCYaK?qZ'
    'qd~Wgcqy-6Z^vKCu>JeEwJ&AWJ}YeU7Zbdi-gSuT4L=*<V&m4=qkC<#H{YySI(-S7^No0!{Dk}=yp5TVyl0m}Vs7urQ?HwPH>qg%'
    'CQQb9-PG+$O4+SiLdR;pdKxUkFH>)z>kW!~a+4CKCwgWw0Y>=PW-IWn<+7#FsZz{9vSjq^_9Z@EQZUFAv^slbL0Kg2BBtW_h^g34'
    'Vk%iLG0W#!R~9&x1$JdY&gD$%GNu>_vNI&u!5zT^(%aVVzM*5-afgH-zlL<uS0eOFlf<OcI6d(#pBT8T0cLCm@??k>afsOtuCuZ~'
    'mUfO!Gsnd?qH&dN8}Fe*md*MK?miqczIv6%4;sv|CW(cEKu+)|P5mjbJ_VXPG2g1zx2jvLW__xeVuf2V$4*?`-=MI+h_ET2^8HP}'
    '<@%d`%J(;LgZi+$kSkX;(v)`u%A1G)tgLHgZX@5$#VX!sVEnJ$7o<?V!60ExitQSuau9(ZYikP1JO#;9HF*N@+qnSGPJb`2SRT9n'
    'K-V86^#@7)K~jHET)#b?%eoyT-42p&2T8Ysq}t#3@BGHy4w7yMNw<Tf+lBSJP>_<yL~Z7*RJFb6&Wq=(5rJYTcZJ8|ETZeo`r2#o'
    'uCZs0zrMHN705Xb0DN`)-vaN-Me*O=Yy|c+#^ionEOGlCJ|46)mIp;%P1)N?u5LHiV^6zzfw8ge=OrC_YLEPKk1G86DN(yG_k`2n'
    'cxF83z!qR1cy^nKGh~_uV{AL)2*SHLuf<9I?|1f!?1s4^g5w#y542<ATew)8N_g?{ppUhoTG~BfYI=i<K`-=^9?Zlu0mtJ#8FgBt'
    'z7>ymHq*|mPxLR0hi&JnCvdbdo6(S|cTL5X&GfpC5BDVPgcqXEVzh&kb3{H#8e|)>ZL7i5QMve&-BjYzMBKyn(L<`x*zn#ea<7^3'
    '+Hk#Z&G%RP*35c^_~e8z9>gs%^>#*}%@k=<!w&**&4=f&U%Y`WKf6lxJmV*HO+934s^(+)lbrS>F-O{3*lnZ+>G)j~g-yP~2~O5U'
    'Y&2y*ohXrXCzl5d@D^viXIz`+HWZUo@VT*IuSg#Ae)=5Bg?9xTy(RThR@yuF6Zr$XhvJF#PJHLW-;vq%65*<mq83W}Hq)4eW8B(<'
    '+lOPx9cJo*N&CY=ce>u6gkf?G>LQ{A3ze?NL#dw)$9pe<qxkQoG2>xvi_ZIE8g9aYN#WUvxc6gzv0XYjuD3xEOvI5&_IqY&;z`%_'
    '#exkZjs+~IL5|-U?}G5mgfVmCJm@s0r<dSyscZLA54YOgmZ@-ISB7c`+QVoneqAPq8r)H{@ocla;_g_qXy{bF7FA=F)*#5$EhJZx'
    '6EH~C4Pi^^Gvv9D>`p|oE3zSQX_IbAU1WWrxE_sewy^1v-cMk>#NJck+Iy0NeEj!sU)yiBv`bf(`LK!2hau~-^|+-=+qbTtv>S3@'
    '+|zIpg~Ls($-0m<S{H!53-o#uap|vb)96@?x*=mD)(-L7+F8jc8eYm(A#Pz$wItuawTz*p&K_(WI#T*YFP`SoF4{xm0*8@zWZoFr'
    '6LgPVT>nL=U68Ej;&xy22QKK@9aGXRlzz-4;7{ybh?c}d+B&BZQM7OT8v*J!QT8f+x!~S6?^H=|D?ko>0q^I1v^gj;haofgQEyHh'
    'x|!`{whAxY>ER8xiv!HOMeKTRzD=C^7;d<#+@^4%cKfnuzi+8v1|)V`M4@Q%VQj%v0pW1dv;1c3Y;_u`i^$Jye?8_>GLH%BAfV%6'
    '(xjcTyF(+7Ey4j`InD-#(-#635knZqFh?*R#e*rLfM{1F#?guPHe%!m6|>9F6YUvWIY}$*h^!!-=*&dk%{pjZ=y7jWlB89IDsYD+'
    '+beN!lz7t=$A=|$YIK@7vGhE6o;!tyb}H;=JX-eoShaJMR7#$qqe92in^^)7vcJB4_YB0apfr3;V9|qY9W=H+T_o9~QOXBYDP%ib'
    'app1L4A^Og%e%>-_jGV3ER~);MyMEJl%g|cup)T<q|upajc8sY&X^-kEhmwsn~~D!^|i$Gw9B+K<rR?9Vk1bB)>GjT5h!4L#_)23'
    '`Vco8VytyI!3lf*tqi^sMkHC8Y^#$F7~WapQ&BwGPdWj6T4djX8AtS$jAV!JMeL-j@qiV$!W5G56*qT}`%$_xI25!9&!o~DC*@NH'
    '55>Bo7@OiH2ZlL~T~5&gR5#vLtfN^_IX{J4Nd`tb7w>v$I0DD(YyxW^lf6MemTZM6vlrazERT*{VFu3?y4hUXXKbo3>~&I8#QXE+'
    'oxPP%e5njPcdP?LGLC_ZC{QBjf@@69G~5H@BLsV}gGHQ!3IXmyubfa8v>U$kB#?kh#$nw;hjoJHnK*(nI-o_4kqxTX{*P&wYrRSA'
    'KUdA-#<41v(xS=fjm6R;q9S(OnxPbyh`r{xJ%pw*4sCHK6KrO*hq0b}B`tA=a5)W;!6>65h@+%~htafjXueLH*@~;VimP9z;##ia'
    '+SjSLo~yY2bt-<4tN6j!sko7=xbbx=eweHH;TKoj3;XTaaGI-m;|ptE>Z|xBfWIQtuL$icg8GUez9M|D2-hnD^@_l}A|$T}!7GCA'
    'ieS4U%&rQnD+1~j0J`KzF~0WOQBV6@l_SP@-B0|^FV36k-KmKYnEGRdP=iCQmBd<QpxrKNr>*QhaTJBQqxuw-H4_!q`OFkZWT=lE'
    '=Mie2BU<h`KRMTN9yz?VL!2yOY9i)pA~d1a(TNorGPH8!h5!RauH)E&zvDTX5jCFJEYQI(jzw)6-1c#99J0dGd3eY^S>gFTJoH_s'
    '=ayUQ8wiVSYC4+w4iU&1n6+?G7!8Q%xiQ!m=4eV8H}@sx0km>{N!-pGIqMZN)72^t@)ZZBVpqm%K)%pHcU~BiWB7}lTbB}QH^KPd'
    'c(RRy`3j4P<8$bJG&6Iyv%+GC@B7SZpu)<vOR}@Vl4Pq7O!a~5rvAL5bt^HF%3+gJf+QtieVa5eRjS%TUvn!=^~EiswP)S1)y6_y'
    '(ajiY;zv1O_nOb+>-Lv>4u`L0s1`zn=yeu}9P%VGGRjV5oW&ciq@uZ!FTXQ#AX09Qz}w`Vo?q*;IvOog0BA*nuc8Bx*7yjz2<))h'
    'gg%jCC^BOY^3-8gmu2KD`pb~gu5asD%KB!Eb`)dIcC6yjD8euDLUcQAg98c|ghpYMBT;j?B7L34w;i#c3)YPAyqC*-b?ad>5~a!C'
    '-QL>|TEDzezyGcUs(Xp&*i`jeLfo<|WHk>bp^T~r!#xooh`(aunoylzGKV77h9LrcwHokf9>s}A;}qJ>?ZWQ%{@(uf{);CP9uj`}'
    'C5*Wk(iwluYC_N}ox-tBm1cdbr!fg~Zn-W0`stllsz>n2^*_F+Ln-gkCPQFJIxc&cQ467@Jv+UWUauL#LoD~lJTbCvh#uR9`C#h('
    'nTCYHOz9;Rmk3InJZD>&zGxxzy#T>x#LC~d8p}itMsS1Kkjk`<ya*w75={!Li-gpkb8L*|x#bqbE%60vCwpb=kk)uQF)^80X5>Ow'
    '$pBX8GWA)_a1)Q@w}N}V*_0afF^dt=cE{ncr)J}33IfTl8bs2R6~0_rFm{!(F~roBviYUlIM~6Rjmz#hoLpedj<tpeQ->kVJ_KVN'
    'y4r#q9c2Pw{cQW)tAD;Jnb_O@5})xbB7&djBG2)kfn1lw#Wi7Ztw>If99)(O-8{%pgo&omfJ52OKw%#Y4%4W*`pj`RT7;cUbY3K~'
    'uhdPu#)mu_b#TEEX*ldL0#bCvJ$kfjZT_PXmW$?mGU^O89@tD*O%YOwv9EY_+Gn9U8}&kHKy*j(G@%4O?0p^Q<^+}N@i^=b+C#RA'
    'x6wNtZS>fg4Te1*t?g5fhyu|p50{N)AK+Bw)#|mU?UKnsOF$h5@5N&QCj)$Uta+Ki#6%2J#L7@j7m%zYEA5GKI6;%VYc!z61Yza!'
    'oBcJ!eAfFwrfvWAHb<_o&!P5knCYqROb@2frZ0rNOeG@Dxn(f~yq%&ER%LxoI85ot0xHHjYNLOMo_T`}N24)&oKLp?hrOLwFZHUk'
    'X^J~&69YqRvRSfZ_xA@r7))Q{O<mn%`hD|`n9z1lY*eTqS~a!mm77K3q@;f7PF1yGpbG`Jszl&&O<Of=_x|8W9nT0K-4cY~qBA+1'
    'Ouc^ykl|p}k2|lDQ<gjONb4|Br6qMcO+TB_(1F>)MK`3`Hkoc)#*dQrZdX;V_dOn8LkkF%bFPV=ZTW?t82olBr6QE?Ul&7!o3`+7'
    'YaU<IfeYOt!txZfm<b&cYb86P7r(xOW@^e$R6i^cULq<x-;l}KJ{q^r>3pHDUR``->^*7h`TD*9VPXirU?8ntyxxAA=ntNX<$*0`'
    'ijA7$J5f`7=Y%)m6*<|}G{|n|!+W4yIa1s&9L<S?IyG00Z7(#(@F+F{qO1P1H>uFIJ4TwDTm~4;N)t+`;@hen!-%adDl^36WBUm<'
    '+0HfN!P-2H;zLNNJx`6*{cjliJsZEB2l2?-zZ@x%j%`|pb%kX>j^c3cLpz8(R@xV`k+_JcGGv+J(D0a6ylKPB$*P#(>_;)8ENY1>'
    '9l?4kQvr`;T&DhRwpM&RnoNl@)6w324w{w*V$`ehb!xSuyWS{p{9h~_d^V=L6<I9@Z;9B>4)b0ZraddUv!lEJz2mokkGlR1qOSin'
    'qo)6zqMl!UA$wg!>y-@Lu8n5>#j(w83p2muT9u{mRrhl(`Fvee>m&;GYPxry5nB2_vjLlVz9V5(h)yNe4Qvh3Mc_;g`;#irGGZ~J'
    'C=i*M2+(k=BWk&oTaIWd&a;G)<T6Llphy`)jnL`H>>k^kn<c>Mi)ag-&Ky{sj?puDx05}s_R8~^xc4>6<epceYVvMJPAhmr@GX}P'
    'x>rO%rgOl}OXq3U5~mNCuIr+6>u|}PPByOhmj>-#DZ**A%?OGn0@wZP`Vq~%0ci<(MCJ+hkmhss&=Jm@+i||%_$luubjP_B+#8Hq'
    '+JvdCWVkac1hbd0pG2_=Ej!@5f6%Q7IZG&-*HZ!$2GZ3&FDX_`gZrz^l<*h691ha7n?MHsO1b=UJ;+5wT~0&mG(<K3m_UZyzNO7<'
    'rhGg>%_J-_gWL&^2U@HvphRUj++J1%lH<(UWO{Qsr8XOu3Shl}g+NA^YY>Wl(pg6BXRA8OS^;Lw05zdR#MxN}g!A;!HHyY2_;?z7'
    'JPX9Ssp%E;P~ldy_L+gZ_zV}w#Bi4vO}t#n%O$`3CNID7%UoXO#RT*Xqmy475<4$^25aR=ne8mXSecOCuB#O0x+E+f0VDHDOEa1P'
    'fmg546_)FnpzHZq>-jj}-Z4lpO#LP=hi}FAPNIIBOmoTfo1H1h#Jj&X0{F7{-(Md6`)kDhHXPlJ<9^>b>i5lKet&7i?@!BOd4Ia~'
    'Z@M1LT*(zMyK>wS-Y4dKnINK#ahxyLO-UpXvD{m*!h%&96yi-?th7xnuA{|Ot90kj%$J-sS4`LnGh%KeEeVjZ%=xg_`tM&}yfF$)'
    'S~Ml{;8^Q14Pt#nPTCR2bDe~&SbOO-(yk;ViJ0YE6bjGCmPI&p(y!twDnr_|adu+eQ*x$*q)x@Z^1{-+v2jt}IcH47L``+BsRJT_'
    'Xd~e54N&2Eo)i)a2>k&JCREQ24%am68+H(*sglGBnFti0(Bm5#qY>mW(ZW;3<0h>6rg<wfo{Eg(&9UzI$5eSKxwWSz{$VwjOpgrO'
    'r2&N~Vs_!2waC=bh#hYdX)mb+2FwVsah_20HbauiZ3fevemE8!gnPO<SGsUg*O8{Vg>eseEiz!N08(_jh|1!7acAFxdt{uDx7S#9'
    'k$&YJZ<bh_bU4a)a{GG&?x$bC{W#2z|Mw;6@;9uRYur5N5hk1mytoQ43!`(@K57pl!wbh}n{4<)o4ur+%T=dSLht#+wIEGi+bGQK'
    'B(DYjml30ux?-M26XH`89lPOSN`ihB=SEA2#N{MfrgPKf^dlW^<K={-qY0;pHohgY%tedHHZ~Jwa-B6FapTjHjZrn4S>AXu>V&Ox'
    'Z|}vc_pPTd{*QMwEhQv(xES}n^`j{pozBL-?=RO77RqCb4>2~$HEd^csER|4ZK!-el(4C~@Jcvk<7VM{^|i{$s1yHSED~c&qEM)i'
    'SuASE%QSM!!nmuSATL6J_qEc|$uLRv_<&{$;}rL0;OB&;EBz)O9?{bfKy-5Aw&_q1jHMPH(78v2*3p<|XF*1Fw3Z73i6q~};z{Re'
    'IyD#SoSvW_qAi@fpO&m&#t@-+Q|-8$DtA>=%pG3E{24H;3R9=WTL%$Ga&XzdrCGt19QxmSkAF*6aCK(zDqQpB4B^*d32h*|F<bZ<'
    'jN#|7hF{Jcx^A@!kyD1_$Xp7?Q)r%!`U!?Kcu`qOsZbO&_uIraX?aXmNo426^O7o7-z{KGwfaar9ukoy+L*3p?sNI+YF!YmL2{QB'
    '$UfdcoAP;vbvSp_B6m4j&vUIa$&Pq-30=Wy)APi0Kp)Cwa2s&wai|ggL-4o$Sf%eD4c#I-6RbAYA9PtU4HJwxO+v2>B0T}39TlF>'
    '1{dvN$O;wO;}ZyfQ8$xQ30xp?0&{3C#Ut^E+@22Lc}a0{mZa$eVZsl{PudEDm2jW@vbVnj8ajPr!JeCA&&lE2dbH|PEBmE$;rD?`'
    'P7~$7bi&i=*#{a*OGNA-HhqVG2cBk6+tfN=z2|Te<EMFyoMANOU<JH7&(%T=3A}-H({|@zZ=ri>Rrfs8Sd;f5<$Q!AnoICNtXg)i'
    'ti{x@qifi4T$y~ZJWjy(s|z+}TH#?7elDDeWj~zMq0Q`E&9^f%L^<^o&3Br`OYi}^<b(G-QBjRp0~?5kK0Qv!6A<6nmufshpS8N?'
    'ViUV<H1Q(nMSt)XphiAC_W{h>)iJ|39f-4Go6o$?XnL$7==XF+GpKdNLRzstXe{ob4`^dYdw$-4h!1>gQ|tW<JfR;yJlBvZ{yn(M'
    '=~ORxrtxIkxi7$yJ~z(gsO3BbaUtz8g^JtW$#14@9`rrxS%Q9bH5a(-+M~6*prtZ(Y5_E>Z3~{Ga-x^5U$uMN4B?a4^X?iH)htl!'
    '?uvx}d7{P5%D0Is?DMaq3rqD7-@UgkcHUTAIZ0s6z$_LI37ivbDAr%HPVD=LSHz=CB~^XUSWmo%`<n?Nj_`5BJ#Ze4`xHX9h3)P!'
    'XgD^}W2bjigL+w3du+3}r9G8Xy4D~o-Qs{e6DbKoT)fDiGqTjq&^Cz+p;8jy<lqRBL|dL%SM$-N7ayj9uQ1#s_!Zm=P-)fzWNlcx'
    'hND9`L?sq6wJangz96hk=b)Gg&>(+^H6M!JltkMZv4_UsJBUeL$S|P}Cv&~Sak3Q0$?ED>5haV}Ef*-uPqc6G0~J}^(UjE(Ycks1'
    '?hHE=hWpmUu!b?Un??mwh|%}vG-#knFB>7ugov?Yf#LQL4wb}tN9?Cs#HKq5;o<36F!$9FduGM)b_8PLK3F_f75*@qoU&#<Z|u$`'
    'Bl>k&Y-pn~80b4vFM&IFx#L5%JA)8$IuGHMe$I-@aq7p)QLg6uwl>-nYesV^L3pGWE>fEXmh)r5+LKE^u{zaO0`!Mgs6(h|B4frk'
    'Gv2bR-TQp+&9AC<of_ssZnAOz7vFMyU{VdZM5Im7TwY$L%Bn<pyl0wx+tSM#CdrCoA?#K4i(39wZKVTgUfpxT*4!E^``bH1EI_p1'
    '2oCg6*cK><)b=pn7O}7f;KZ{hZL)%T-oAMD<eBdF-jl`$Le?8UHY$X#f2;-#t@;~<Dc<-YPE=EAK8!askIFhYWKBJ6urEf&2}L8='
    '+kfu1TYIokN5i$oMx(B8!7it!`_c|IsbNuv(PR99IB0SoED^*kKdc1}{|W%nb9hhHD?c`x?$os-Mo**Zrp7-!&%84=8p_nV^P#mX'
    'UWMohvst63UpcM8Pe=pXEl{GzrZr<|Gcf{lb?Lc~DF6}(VK{HYx=fdgp&MT?tFM09$M}LqrCA95{lS6;B_N0cM$Q(Gj(_H7aP$^b'
    'yZhqT5B97*2&zrDT2KN~qoqm75JR}<Kp;_EZfeU6I0&*@5;oQ}hN&~0Eja#kXb2W?FYS}qg1Vn+47G+hz30BIG3&M+U&VcK|A|UG'
    'id!^8c|jL@%j*a!sHvNj2iF~a_ImqQ!+o2yap#KOL)vylpV%WZdPKbMN<GTmc%z2L-bx+4YL;=r=Ldq`yO8pFx5dbWHUd39`}GBo'
    'RHM#szrF)hB5*9f07ido@XJ%2p|uUk_TmkLFYn6CmiTY<ns$ONPxQGI77+Mv855oNk_8pfj(NTN;tefs+gM3r*CgENHY<^>Lj??3'
    'hzbI#Lm(Z%@vuFMh+@({*g%g=6j#SM2a)2~+yN#Mti(k)NG$8Jmjg`ndB(2I!~uoUF;3<2uuWKPuY5%30MeA!MOz<47t}NyHRDK~'
    'kBGVz)gumx$jCPd{?qPGMsaNUZa@vLp=<$ksk45Insp<_7&}>D=GVLb1JPksTTfi%4WNg;vU$<gdF7Z$FvKW6xDy<Tm7co2HMtFc'
    'g!DuYN!ZG}JZgP&ZFq)MA6KVvrbOTM`W-`qS`>(?vx7{x@q)$3e3>SLrZt8a=5z+CKE$GWw)Vj6Xdv7aKXt40sn@<qp9EuF)u$lW'
    'CokCpe$p|(dN^4(4nSh27!6(E1?4hs2o28xI5WJ4Hwm$<1ZjNHQsOh{!2=?a-~lA!T0!_pei-w@0uk+XV%pi|G98p9H6?61t(M1^'
    'FcgA)nL3R|ltEKT+gj90k2_2x!#7xaJq~HpaX=)Fniw<dk|h<lRe-S8LG1zksS~^f4+yyg59#a;HYlxHso>8;xf0>FsrJsOb@uSb'
    '4UFI-*r+aOMzy_L2XMPqDc7lNCuV=ZmGX~J+l_JN(@XI1j-TxXF=c2cs8GfLakT>k%?=O>JE$WKvk*M2TbMFYm9Q>S2+5Dz-O~*('
    '<+r`NUH@Q12Sq9<GcmU*Mb<#tlOARI9R@AA(!SS){pwvTDLXj!z@QB5jX25UccA|#+gefFEW@WFGUUmQn`_6(RWKT+MYlP8oS6z~'
    'jlh$??c~4h_}L>ucd&Ap-r#V;!$XJ`XcbM;PBza@l1GzSyjMg4Wm5$J=La}Du0g1LTI2FdPUn)5qB_SkK0;BLj?|yb;5PLPh}8hQ'
    '-_5z@g{X@|T~c3WLs8brfaAMozr1@&+&Js5;%>rhZT40W3mc15VmfILhHN>mIX<nN9fmvxW-eV|f4;r{^A2s%r0RtZT-u&OJ+;SY'
    'W~a>_vT%-ROLAy))5*p+>M@}t+~VMr#hnN2VHkB}hw=8EJqGb%<a$-m+8*|ir+HqnjuOs|6|5(wkJ(n90;yKhb_}zuS!=mqrUHZb'
    'SeV^5vHE~qMonV9oy=H&IE#-7%XJ3RikbBi-U5SCv(4{5c|lZ`_)VxP?^*3(U7J%<zHovm<w`lyH2X}{@&D?|L)lun{7}VnXx?|i'
    'c$z~^1jN?hPWjFV+!1TVGk@vS>eXTFFE-sWt#7sh!sCP$H66kx=07%4CU{dlc}Sdqf4#;XO%UsiI0mg&Xl^00WV_H}%61V8QhEhe'
    'a_q!Px&Uv1e7HQU<gCk>==JMu+io}W=~3{QThfjzt{Vo;S;aH)>w5lyutS+B<B%JtFM<71wO?u_D{v;l)ED|d;Dsu~^hA3=z#m#8'
    'pwBXHqBdw@%l%rl45b79qdtQAx<f7Bp$6~NK9wkij-oqsI^7}kcIP3b+}U7fodN&90x~Uc#RIynf7|OxXdXhu>1*J=R^8Xyp&|9!'
    '5M=U0U>>UGp@xSc6~W+r<A`MSrT7n8iQxK`SaVlm&0UE#cO}-od?h|zwQZ_@`E3(4u3atqmtCumSgbogA{S>3uzsLIzvWg+UIVvm'
    ';Fb;CvccbV8-C@L{gzuZc@5mMfm=3k%Ld<UTi5>b+qzo6@!|^x&Mf^$Ui&)%atFvBJYaxm{CT%}@>+GvR^75yw`}$AiYER5Ncpp('
    'j^wrKmaV#Ft8UrqH$yx1zdYKhRlfo-=>sPVN*@`P?gS_iph)l|0IXX1&qPJZYt1cNbIaD;vbDb}8v8dv-hU!$OI~Yk*_vCn=9aB}'
    'Gqm{NFOL@Mm2U)``hd-X=0}FjJ5{b7WvVp>*!p)=nMIY!Yuzndcgxn@vh}|!n*9?&@^{qhqTb}S?v|~)W$SL)`Zq({jlVwH-uMH7'
    'T@w^pQ2)rVey55uRg|gnU1@dWJ1YO0s62Un;Ff*hmVMxseekzE2L9YT<NML@^^S(*^?_UVfm`;0TlT>>J60b4<&TvIjeqmz(F9a>'
    '>wM(BbEk?jHI%9G<$taE{iyk~95u;n!!6rz%QoDyjlb(L^%vY}-;b_8$I+F%Hr%oew`{{L+xTY3-o{`4*lX1Pgqu+ljM=UDk@w=A'
    '8p_mArpCgoTI2gs_@8hTCa(|OvJc&|58bj4|E|aAf5^`LezgAekJjY%p<DK$TlS$__Td*E%P^dnk6g{`SX7ja4XA`FW5sFXTH6@1'
    'iXr;+cruzD9b1KTxS{>m+d4{3g<XC_$D3oG!`_(nwX+o(n{l>FmI6|h3BAt7#~iFyt6Z0KG&AT8$80E(nwuLFFjn*Cv1&IAW@rdf'
    'CvnfTo%8&7G}Q6dsAG}VG+#*VsNk(uwr4x%w)U)5tZBdd;@#f9){RdBvyJ-Nfk`B$u54bvnb=rQj|n?c9}w%y3Oyx5H2wfEaq~gU'
    'zA?MUqRD;6M#Uwg%P!j8qVUg-);sO)sai{{vc-mb+`fcOxC$5JQ7mEml9-Mh@ha+-)jPImfYN=az2fNJ%sT5b3^6>|R0r|6q>ch#'
    'hJ0`YFZg3$dGQ#2evT@eh?+GCdB|{}Sk|+-qiuS1$SVG>JkiXz#+^hH)2jDB7`J6+8vXJrwi~JA9b>$4q}{$A1zCTlGk9C@yd&*{'
    '?D1n&XHyz&yKHsO1kU3r#{#l*(_?M2U~IlQt7RqPm~`B|3kyu+5ebdnRlx%v@nkrKI`f$N3X?E;+@!~hacQE5FJ=<U!(?jf<Zw8e'
    'D=!Meo-u<@>|y-EdglgBe2$mc8##b=%sp&3u<&H!W&O*_9QjtfOJ3|)>{KITMU}?R7>hfyjL$L^*rrxk3)ouGul8AHOmQ85tlHh|'
    '(S!pwj5~+e0+}=p;(*Pd4V(l70e%3zqrlJl<(zSBB{m+a(@H;(DWjcR)!snZpNK%eb|JDnVh+b^(6x=r^PQN3QO~#I!trN~%@7-y'
    '{kDkJ8E*xGJlLL<=+9A%zme~_nn8&z-cDQ=+0p+5KZyH52TSC9(I99qmByCSUMcAf8zS$v2JZi6$Vt;y($Re>tLL$Ya!M4tD%&qL'
    'hY1!}5`JpBlDJ7`8~-&0Hzk0Te=&7csleUZQS4h0)#GWPy!p5q!~4cmy@m;8qpra>f}O7p70K}s*_EyNwyvriBZAIHxIys<H`cTf'
    'XJ<xurKz6P9tP?@eyV=9?^Q+@W!)g`O4IomM;w`Q9Z*|1Huy56On3AQIPP63{eChtG~ct5dXSDh@h#_A`7pS<)z9Nh5SmOBrwM!<'
    'F=W~f?mSOB8%-8&jedON(l8cJm?pb+F<}~VN&ho#nOznKtYtdm!tS&+UDy#u%N<&*7VMbeN%~HpBaS*2%v6uBi74G5lUr#>Qe#Q|'
    's`|6;iMZaRvw1LWw<GoQi#Pk5a5;fKOs3lQj7gjr<PQ-HaJiY#SQ3y}C=>=6y~S~UPR=OztEw80h)%lH^v=i%yq$tU$bK>Pgc2Im'
    'uj0y*qx@#wD<&Kj)y@asi7}v*TCj}{v9&{Z*Zlo=L_33HphsMJnr+R>hU2QTz~-_4n-R9zM0#~Mo1S*cT~I>-mOHcIX}XBR8SK=`'
    ';xeRcR-HbK&c?%VN^r8+++-&s<yTco2F27m#{9^l!-QQ}y@NY7Jk^U?C}yKv$k5)_=dt5$-PeRRfp`XNN~#dBC@;--2B?gWLpyNQ'
    'SDugC1V=x&_TKOAzW7z$E}Ra@^2^(2@3!AORkxST%r}@9Zg~6M^UPMW17ZOeuC2}atT71dqdxVUg15I+&~zL{vbB6_Xe!<yFo6;%'
    '78^}_yU}dRro~Pe=EN`=+FuQa`I!!mW{BuRwCG~nxH2T}XFeHBr(wiK^OsH#W?_Ih+IbS!7lgErIOB_*#2sQ%hv88SSChn!p$UA`'
    '8WWMf;J|G=MGkS^{qg}$H`p!68oR23CjmK^sVGh$@rBjr5V^odaEi<ZzwrB1NDnT)gdWr}ak6xvq4mTCM{3_IXdFiU#;eXlstmlQ'
    '6YH86499M780O80Z<!e8tJo4-e8wW@6oWd!w@C`OH5GGiM7O;+`!v4@cb(bJ7F~|%(Nr+zx>jW#Sw|FIhhK!@FdX`~@}8CdP#QY3'
    'atjj_U3Rzj!<CJ`$o>bH65fly9N}$!Z?&R}#0@gdhtM#ZtX}LeRlU$2s+~ZzG!xjJTWaHq>P|3f)cK!Cl^2uVzZR0-;uUK_BUl*H'
    'yDG$iOMNd1wwe#Os?78=bqP23w#Nk`G_5Fs$tqZGlc*4(MihfTCD(=VKWO<8_QG)TztUuE_{9-i#c2(1eiOwN9do0&66T;2%W_!E'
    'PW-k1g^bI0<KzfF*m$qg%u7UcaZBGF4QFRj%w93*Q@O0xst+E>iB@$l$KW2cp8E5OeYuI6PexNlj%U`S-)zXeb~<E>j)w!2qLE+A'
    'Ui`}J#Ufr7<C5LU*%5wu@!6f(?2qFVxEqBe-B%{!3dyz=>9mDL<%?MR{rWW_(oQ<o2B7)9r}Q)uEIC6gq}>XJEyUU)CDwig#4I$i'
    'wpdQAS%<JWmpz=C=Cn4x=0Le5<W-6JrjkonZ90b|yY|qRb`tQwB7@K{5rhLomjRo$;P?rNlO=JjF>GJb`KvE@C0c^Xt|=JQh$&{u'
    '3hn7!66a5ob;Vb0GSuZ)c-KdVj|$nnw8%SQ4uRStxNZeHc?a*`ICcM3t#1;$+U^>k#Nv1t9arL(SQ8)N4IGlgVcOFHT=i!hlRIvI'
    '-<8Bst{z5GNP8+j50+*Ycff7|LzA2sT0|3F>~NA@*&y5$;n=)C8-X17v=KI8;X1S-Y#DQUFB}tF5>g{S@4ObbwrV;VjG3FDBGM?6'
    'vr)L1A_@1w@B`Onk8c2mvk?)u?GFAjjc+1^N@nf`BkoIe*1n_>4BFUxLAkd6P|cWuU>{bAFxI$BVponE*mXZYrM-&&!e1`YbhGG`'
    'd#EF^>dxLy*Ct^gp6_Tcp6nmU4GqNsgb*AG#M*hl4{KXg#GJ?yj{4%@`HqetBhjZ-Pf}KlWcz$ahqBGb4uo5KG^$PwuIVAJ?UssW'
    '!{K;3aU+qUlQyxS&n$Ds%hkmrs?kRbk!HcQ%7jBEt*eAAD7i6oTmoUTJS+|S{O9;xt`{gfAcP)D&g-KI;SGv&DXGG~<mo>k2del1'
    '?#?0(tU=vk>IW%si6${%nGNuDxEjiGmwM!cr;vNCmv5iFdtKX9ryJU>5VpK2kBucr(zL3@SsXbS;`MQRFkyf-a+7`>5&=SdtM?o|'
    'rgL_hi8oD+@<+Ij^StM%Ho)a%&1cV$lEN?a2)B7l;=b4&Z5HG)91n<lnwElj&*b9;u;!oOi15IHj-Q0+KZm+&#ucZ%7C1|<4y-Nz'
    'tr>4&xsl&17<(o*&4oY^2jya@@mj56@5sXlIUtdF5w<mx(x_)ser))wMYOr;*0#1Opq?g31OJ5%PK|oq@Wq+n#SAEX#5%DAaro*R'
    'RCalYsBxQCJYK=^d_J=n5sttl1c<(HNql@7&h8{RO)^pP&UKhPLWJp#&!CtrjycJ*PN9xJbRLwHTWkp#37?#LWaXpr%n*)5ez2>~'
    'EYV5?UWOSt(S%%x5FmvF9EKM*v(y=TDj$Y@I{qi4Gxz9<!=p3W4JC=HlKM^MLr{PG!N_=pq|$E6Ot1vKFi}rOg~8Q0T1{pB$L)(y'
    'ybBuAxJPu}DdHywynlI4#5ph*01%uc_j5FA&WpWHlV#oN2^fi6YFrxJ($NYN(oxh9`obrf##6C-i8(Q;aCU|e;7Jg47R77-@O;;w'
    'x=J^Skw@`cA}I)>d9Mu66oTlm)8z=FAxEJy2|WWz1V>=psqemc<2Z}^=guzt?O}&pcKYm!P_fJ4LBwcTUo>ZnRLqs@D(>HC94&^&'
    'tDT^Vu>C}SCeV_SldxnHl0+cXSBx%7Bwr>b=F*UunP7J(ZnV2hu)9pCJDRz@`QZ7h_mI=H`Q}lG{T(}UO_oX6(QE>Og%C+aAVDxi'
    '8+!e$eKMK|qn?e($HEty7!x?IbOo8PEc{+JP+0czs>3;{@5_h>d_Ye?1bnRs&(Ty5i6ea30lt=n)yf6%>(vjA51pWWl!9}~q%H{='
    'hNWF0#9lr+GY9%x*z7P!D$EQm!iGgsQTH=6m7YKLznXk%)OBV|ET+RR71IDqLj_(~aR4oL*$$+FU9Ru*qv53-c7ROqY>beYOgo=('
    'Fq#3Q(z7(!17U#nPy`bOtH+^K2Pt)YJnQ#y-+bSoWxp^^dC*8g(QyRIN)7_90^yw;2;2EMik-J4Hl5HbIzpX<TJw`Yj~vu)^^zQ5'
    '%})YBew}^5lW_&~8;LSu;9_{p(WUP8HvZs!N4S1c7=kVbW@d-YE56_Sw$1D4bwyoBw)C7ST3ZZyZJ4~`fm-8genCyT&zUY}M3V^#'
    'tQS(w<CKN!gwuMfJ8aK-p}v*Exk<-I&{&sS+w3M~P94#6Kv+QL)|0R`*smM<rq_s(`U6C6nNGsccY@vXJqyR-<x3QLasUuOX+Y4~'
    'Y;D1^6`4H=kBRZi*qc&sn_k)oZ%e<OrW2M_gL`mg?LnsYT2gyZRQuJ@CP#=b(4?P7go^o>$Zd!3M=<$oViulLXGRvRP)DvU64*uT'
    'IXKFNLCwzjxE#lz5f8G^M|m{22!=9f915e6ugIF>t12^rOx+5D?2(cH^e4vw@5*rw6OYMP2!}PHONl+18kR5g%1o4RW#o}p-gdI6'
    'cT0WPe)qQZ;_0to3u%|mK^gKVp$z9^^L`frT;|i{LLjI<%wK%H-+5)?Zzs|~SHMMmWf>Rc)S73Uk|>|ON3?aLe6~b_40D5^x?xEa'
    'YS5lITB2r@i4*YFTzF)ru5QO@$3S^Dzm#npm(Th`>YaoENkbRBWg!qGp;Q!6dN#ESbUSzfQ$?D<_Kl<@DBw1eVHhSm1p*5$GXoAR'
    'Lf!Cvo(}lMnga%1+HsEDaG-H|0pN1}O?_*X_*OTH4?o1FNiFg!OXxQO1(BV(lDJvt9-<CF5=Q{HN$l&e4=nGB@u={9RsH&lw?rD}'
    'QG0x^^tmgdBmf47-88Xms1>kIYY9L|)Smf~yUQ4xJmpY8w#Rlj$meYraFh#HL=*{y<(*xtx?`)#87g|~JZWhQ-qYPg-=wIp$K$GL'
    'MIYhqX?43{42^i!+D84j@aI5mk?`n)QIy;wNKssJSc>1$qP{XnmA^~O^@r`r@X`R8)`)K!D9BeQU{PUt7A_3r-ir2tUsn6bE2b1D'
    'vv_*Tx&&irZR3I^Tt^TbLefX!G~FCT%L@T-2`xmzGeNu>+_Lyh!x6}AIaU^fi8K`FH)HjiV4uFx9HW*Z2$63Lr*?0O1lH679ig->'
    ')|F|aP)7Y_s65wCi+OXqm|Qs$Bs2>Q+^~LS>U@hPvS<=bkRHf}?La+L9rtOaNrgrYAoyT@1X2NEk=FF|Fyd{P>HX4Kk+YR?0`HhD'
    'r|E)u<0~3`_cIIa_Jo5%+tZ}Gl4)LFJoP&ok0DA}g2xg<DvQv3H&)qvDuJ^Kq&zw8lM*w@TR`|47Zbsv>m6%hC9Jb_FrOc^kdnjB'
    '@&!~Hs+#s9qErdCNUlv26+s<DuZx=~k!B`xOdW3&RZEMq$2;NiK!<ruZge%1q8spX;z*1$2{6%vnsSO*VjW(q57l1?N_B$?%^poF'
    '!a(EU5$cl1H4B^Y3PGA!U0rlo&50MdR{lnf_(nnk-oJa~EfM)h%f#CD`6@_P;frF4?|;4CqzuBnTvx8{gpu+{E{_>K=Z{}}H%81>'
    'odfw`B0x*?@1_3zP5;h6rKRo5-bcg>Filayl`dcB>VAPqc6gy22ENB6L*+`^wrXhG^d|O8(abldiI<YR%gCz?P5X~O?MOeHfYj0='
    '3BAXHAH=O?tG<I6s)kgA$Y3^}oB2@keY3xB(B7sMxL2!Y2t7h0V{=UuW36d~`h!3jCWjo{D95&ntN{UJR^nf&vo}+1milM%6KlJ|'
    'IMBb4->f3=Rr=JwOS5%ez$IVF2^%SiCFf4_7qQ8i99Of<8BxpUbI^&#-f7V28v0*{iLK8lZ_t=A|4_!crlZu3+vhV@VDaig-J*aY'
    '-4TYCNPI+Wq(h3}WjYuS*|hi)r=fcBdi$yJc3*F+@nFnXW0{_?K3t3SYBQz`1<KAT#h`kYD+B4M(byN+CWn>c?yWQBqD4cXzxA2D'
    'sm9X<;$BMxNOOb=yNb+liQ_QUd^w`sNKbZ*r%lU@xj&19QPEN7IVP70C;9)k_pQxMWJ%iJ`4zPtF?0z_Mv{ebOW+P0uwff$UPLc+'
    '5R5{XvJuFVP)U~DJ?wAad@}Rgs!A8=-u>{#%tT|UbZ&JndGh*XZ!v%{5dU-tR^VQvVI1i+G#KMicX^jGN*wHD_(WN{;~Sn!9Ghq`'
    'p#pGaLiWH|vPRjw_siiRsl0SqrgX@4eG8W*Ea!XFb_tiy+krEA_X0=(=WU*Lb@KwN!amn6&Bn~E8b-hjrF0AVPb|$~UQ!}5BevG+'
    '>r#zMl|dzlZ9z`!?pX7HPA7aJmt9nrQnRMbUTo8BqnRI5b#D>&i+-bJfW<pOX3_4LW=rL@?re6`G-*S%Tq1tfk`Ppab?g7EH9|{b'
    '(pwB&u(3Dj_`i=p7-89=bCbHzmT=>9xYue>WtO*t`mr8^vZQiFnpTfQWN6;BUv(h?%)Hu+)2#7YAjVCz*1x~~A?62sDguvwrwibq'
    'c}}?AAI)#3B$099W|PxIX5IBU*oSDO7s|NGJbEW}*2Y01#?uRbM9pmc<AHTQr(c7Rf!u6B0obqRgP+z8!niZ98^k*^h{pOsxMhU%'
    '`k=~JvOX+%&9a|wtogAymR;Lnjln==21)15#G+Bb@zk^893SJ0WQpXt(5iXa(w(<oe~HkKJL|lsjPDvd+0*@^Y!a>&3H5ZtGQMWN'
    'z)DG5_-Y|ewkbSNowo*48Fne7;eK+i!0cox!>9UkL$JRn9jiIBFrvyBau{yXG~J@*hL#u2UrY=f>__U@O@t&7<0$6osDj7Rn+qXh'
    'zZG~9(FvV<B!e*k49ANL%(@pK!MfyBhvi4sffINSlYY`f8e!_}t=TD__hu>jm;U-xeubZ%`?Xc@LhAZ#5DCOb)h1&n8H;Flk`PNZ'
    '8xt+M%aTRfKSZ8a@2waUf?tid4k*=d>^6lBm1}rQk=mWYWIVl0m4HV@_EIny;T~Dz`TusymO~-4>yBzap|v^7>ZLUeJ_R8WXIkrX'
    'i64&RJ6rq$iI7ib+0GD}r(M=d>-u~L=8R5!b#?Yg{!d5$=@mVFYM(CF)SW_=5KC>M6bJbK0tbq|HfmK6khN9sb9jvOq4oY{+fw?s'
    'C`n_|RJCCn!wObKgvJKy;GD+qNaUokS67W%qsqfzMsJUv98&E=e>7;Iw!VRbMtz3{>e&XKUVY;b4zevwNtm79M+?>aI+yEO@FR8I'
    '<Pv|=Pq*yLdS;&#P{bo=U*4o<7+s2<f&MZ?$==@RGV6n22sDZknQl3sF-wWEFMy)-8U@j}+|fbcuk#I*KSq7EMT*L+wDu|pjj5wc'
    'qhk&O?Vr4U-gH#W=-v5hyQ}0Bq|2vJO`_sS5`Y3R%h>REY}VG>SG0#FnT<LNna2H9^iTeKqksR<zsq&e?5v61gU2OW@jWfFF`wRR'
    'vt-ATJf}*Pc%@naY=s{{zR^(b`K&aE<sc3_bX*Di(hzw#0*@C<*%JG(DEr{0nH`P?$x5s;=7K34g*OiMsKwt0I*L!9);Z08!jbxj'
    'P?a|OuC}a~xLGq64s`&^#JKw>*QQ-j__%&b1%l^i91>dBrar0ZlN<Y_u224;C-}Sc-J)%qvnsS=^dCvfv(UHyOCu)fJ}eOURzn(*'
    'ZObaxL8MGbELoXJ8s9q-hkIswm?<J*6Mt39`|9-z#3SP)i@z)_5T{L|Gurskv_Bk&-Evs`MeT4!gX*dpSS9Kq74OYxHv4QFqXYA?'
    'JGzEG`00z^p4X%Or`68WDs7Pcr?t-08V#h*5{p{RgrmViF&CL<gbG(Cy@VP9MEX<uA?ps5FTIq7e86i&6pYsqe{YDZ=>_3|@h1*o'
    'Wrv7GYugC?+aX=r`-N`A1znib^uORe#Bs+9jx9YZ1-Hh%N7X9+TV$vy6NroVbi`;ZAKriZ^g(Xa9!+|E6l0Ij0Em&x6sp-mA7gAH'
    'F0{VQy5DAfO`8XPn+Mr8zj|8W4jWeDV(s_0o^Mb%k^XMqW?G}LSEve_7bwmYIu?q&W|JXd2W7km;~Tz*;?jPp`Csb(7r=e22Elw~'
    '_Eo+k?I&EQa(-FL`Kd4GCwhxEa5V@;xp-vELmu_hdve_p31#-L5_$+B^g-=7d%@_e=hubRD#u=wMd;}7-?nlEz!VLW+z|1sOe9nl'
    '17}mZaD(^W?QWB}aCDl9aB(`HjbSMT<F+vfHXHkLGW%RMwl8@VFO|E>Og`*e%r>=Vgk<f}2!Tv2O};%>&k#dTm*6=D`>Y<in{zu6'
    'gL{`m>iK*m0S!>mS?it}^qpgiI^H&;z3TqGO6?z|TN{p%_XYrs5)*34IcRw%oavyzwb)LviMrm~OOr3G*KbL{;bju1^FgyoL_X<2'
    'ZS&{X(q@ncw(h3vi`K%I2T@uBRe)m1Yw)5=2hz3ZHgEOVg#dl?-a(}xdL5A#9CuU|x=Yu6{h+ab@1o=^&`gi-w~V3LTC;dt`)8x;'
    'f{O*iQ30QN$$Xh4V|Z`CrZE@*dT~5imUVT9h;V>vHa6R_<cywM&C<Krq&Eo<92vpZS6T}qcUX+^ytFpl$g>t*H-Mn42BSYi<lL{m'
    '`v&es_5c^%@DGe&I-VQ6Q~MQor$v;}m8gdm6aqCgeSsfj-yU_bl~pMK$SiORDo36>60tDq>Ernbg}ynpd*xdhNQN~|JnE(bIbX~='
    'f5!1v-=`8AMfoY68NL~bpQ(xW1w(8d(1W@cC6odYB&~ie@`5SA=wSavaYl%X+!srQ4hp&bv~4$Sw;Y!vUeQtzfD=qtjo%WeD)e@i'
    's7r*GA&=t1K!<8L3L?p7iK3jC6Smc>lKpam{u|tX*6(qPw<E2NKjOgCOHttj8-R9R+PzA2fZdT{6TjO9m%I8NBc?+|En8cp^`wh4'
    '{gJe>nByb6PQLuF|9SzuScJFX7KM(z8$CLC{#?|wP$s5o@^Z#5&$uMku4-u~d!mV!n-2bzgt=9o=#AG8Bj7khAQ*-f5s3bQvEs7U'
    'YWY5F0KK87i6QcvP;*4x%CK3fXo@-HkJZJnEVAV<S<Po%(V;`hnB=i`x*@{Rt5a~0wal|(gd@BUkkq>&>g6S0lhg)nbgFSoW$->c'
    '8C^Q9{N}(EQh&7GEa`r3d3?Y+AfT!P0%Fo5;j1{&e8r>V3b1<|W!`Wz3h?iWSTDmd8gyhsN;A;FnW1qlqGL(|l}OT#seq!!hJfms'
    '_246#;=MZWb|x2byUgwa3x@mkP$fz^%qZWx%MjBK**Lv)Ih^-D!e__x@#%vK)SAZC-~l?U8JZ|35d@tbBrIWvM*!F~e|5B}de~I`'
    'FEn)!Hg)h9nrehiHU2_V_rj*`{e`9u!=?`ZLR0s{rtW`VQ$+E*i_tu6>*#yhlCw(Zaqpqw1I*RGmgoN}kT>bPB5KfN5a7o(y}(w6'
    'iB{ib0Y+3u9T0iqvFYsEczbpw;LDEA$Fh$O@#nO3%Yxrkdf>t4%)Ry*d**=lF}@>a=ipGVQ3cvieTG(usic2v2lWzziXS$(6-A~='
    '7d-{1q4gP!0OI>`NQZB>U>ruK!(I5~4}8T`#L%DtWRPb_>@LRrk3o1e@oT>kVX3P9l(-t1B&KVaqq5~1<3Bzoy;=8j)c&RO=*g>R'
    'rCdX&QLRzk*3juL@aCx1R~$VMcZC{fm^#;6(|(2^vlAb|3mXS9++$VYIG2omU(o63!>Mu7bxI)B0Wn@ItAGqC<V^wB*oQ)tsd`Iy'
    'O8E3touH|oJy@))8jmSA%~f12oA_?wSr<f!&AYNjPQkQgo{32l{IgXQv+kJp*Is1jzGV74UDsi4Lx-n|7O$sIUUOXhp5y$B(l2wq'
    '%S<`?b}C;_^~kEJSq2+TC+U#)^2!7!&?YE*hVmt_f~725Xs){>#b;UVgq@E@I^B6(5YRu?h<>0eI+rTum)OnqQxMP<Df*zX-S9D{'
    'tC+?kLCOsBC9H8ub*?7FB1M(sh`8YSO+cGkGXEr>55e=N$=sK5?tmm2oJN^y+Wq{&I1P;^vS~lUz=55Kfid1PYqUOztC=1E;PyRa'
    'Kv0bypjs6)!N@E=4EGiL=Ed~W9yQI(K&>U<n|2pbqwlF4!-3FT@kr=<OZYDy5P$ue_g3qNFn4x8V*YtTMiU6hzvW5^h2V|R2OEbi'
    'dk4+!MZXpJ=K=3%MXwkRni_}f0Oy~G4$}f*saO21g2&X}qM$9-`k)|*mcWIEpC5CI_aIW?he|}Jlp0J+{xP+(I`7<6{L@)E#;#UY'
    'vNm}YMu_-~;-3)4(JQf@gyl0HnH*6%+L`nTYe^q(vu#_REoAcNTm>eTpl)XpOW4_vvu!pepY5i8!rWa}4Nq@QUAxR(qC2@S-J0}i'
    '(%d(}J!3f3K91cvgm`egbP9usY3|%X+!tGIwi~8G>riK7fi$oDBGGqY;(}FVDQ>CEsuFQmX$BE?1{Ph27>ULk`jaqFi=Zns#KP;U'
    '_I6bXlHIffIIZ%L-uFr4-hGGoToIBg3=K)DU8c7JegM1KSYC=a7;riasFr%raTk+?I{S3snUr*AoU3JG!nCXV`v*pM2{=xMFw=l8'
    '5sEP<6)r&iq(s&Q`zAH>!zyH&g8qz%exD&5h)%`-UAtGUh9);pldK7PbW-saf{qxTgz;p}F-UiYBwT~ml8KDj0#XRXE)UR;lLlCU'
    'E${|q7s?YM<ovmS|Ih%FPgUhWK5=x(K|PNy%mA|0z~RXDH0(+*!=4K>6ENBbmLuY4-RQen6w^d6`b$iQkp&z9#+Zg(wSo?Nn{~pt'
    'zv#*uXc+w<8%Cd@XV=}x#>X$F1>x~2b&Rm=>)y74Jrwj+7vgjNFoFj>-h$Tcy4%8Q;8U*^m@9(S$T63(Zk0W>Q=!=?74Vv|?x8(&'
    'L4|r2oYvH~Z&3W62Nqmf8`@O9XUl)h+cf^n^NliwBGyrQYdO}@E2`Y1De|=dVB9;t4G%>LR{}tZ=pne6a`gGLd_!FB`ipG@4+(kw'
    'rWuIxT(cxKDqge&{|eZv>}3T$F=cklyRb$)MrMuV*W$}-wC48u{JB(cOWN6ffS7sFt>L@iF7hbDS<)L%R8qoFOCBs?q}3x;T)sU$'
    'mctSgR%W|itF#MCuIj1+?!<5<sd6RX!-Iqrk@bCsc_?D<K83Cosk0<FmZKtpLR;C_Pg}Uxg8o-Y*na`fUy=aKn1(9ZyzW5oa||$e'
    '#oq)2<`e5PIS8f3zJLAH@uZ>#A9ZJ!$m-S?5#eGqnar#z<1VKB0S!vGeBaWRI9?^P$+1U`0-j;)@DFWFs!Yw;{n;n!LXQPn;g2mR'
    'g7g*o;~ZybC&rD(1^B^!(2Qn#*}`0c4x@IZ!nKv@?&D;rwW28Sia(q_JKt`$m@URz%@#R4Gr$e~nc*$Y408d>(AF;*4r9?vjYZ+N'
    '(+>6t{W&3V#yx*6&*{%ehI?r<g+gG^4EZv~XZ-yPP-4AQ19&gGb7gqtXvY#0JDc=|%0WWyT~f}GNW|yiczmN)#*_nT(}{z&W;!}l'
    '`z4!-aBQp{>&>&5zw4%<(H!Ic=yc|@?$qRrm=CFOaRZAvd(;t9SFy5gj7~Tw)s=9I#2g+8>QHJTd4POK|B<k6-^+Sq`JIYEotC?V'
    'f*oNuS)~ad0i$}vvSHeQodSuENxLiG84o=@Wwx*vHu((<BMMy;2X>w1MB9AbwCQLKZfO6}ZjHy(>$%RG=D5|>k&Kejd#Xi4BHmZ2'
    'Ic?nv$4H<)d?UEHSY;m2Y`~~+97`>F5&BLtqYS5{`=vjdq-jO9HxNEXqo!n<WFj+J%oV!-)Z1E;g3;QZnHuqPZ!_@{IyiOKZ9uJj'
    'OCeLx6G=HE0WZHuV6WiZQYJ+J?>4!?Y-m~6b-G9sjsjs?cUo0e{M2RvVFJ*fUeI@T>nOlj^p@l7)Se}rOUGvSL4N3GkI|;`J?s6G'
    'pK{r8a!&ay`%7$}VyUqzr)F8`9f)c<3!1!24+?eq$v)*5!qI#5>g)_jv6${y$v^_1&GCHOrmr!mT!|{S%*Ru<&FjcH{A!tcJ6|?P'
    'xc~+_Fy0==Oa{xEEdhTeR>&MsuY<p<^&&rwt?+QW^`s(AHtw>FDc9AZiNo>3f2s44|9Xq93)W%9#MQA2FdZ48r3;_{f1JP#d>|@P'
    '4ev8im<VQJ`)rqwBo}d!$kZCb>f)acE5@0eO(@-Ilg<c^(nj`8oqAH@dKR<T2mK=v&QHE$xq|3Ak0CW0?3*u|JYs)w*s!P{7V%^B'
    'Lws$L1l!ow+iSe^amUgcaJgA`S*jeT=~OW%^{y{QCZI~!-fml0c0(`DpF;_kT%UYhr2_WrVv7)3IPC%#f0VHI6ffplhY4jp#Ba&4'
    '8l8?<b~<9M?{wmqZ=RRIE?rHM&IRp}K`iV?G!=rQ97ti5L-w6ScE~X*MZn_OBVyXHaD*^P(=+Bq>-WLV)#?mCia%T6(eU%*)AP7$'
    '?wILpa!D_B#aEikUm$bQlmh|>`b&*Fl?)q!^W^7bHcm!}a9zx&3tSJHjj9Yq_GHE}(1{5l?$3~Ffo~w2#&;;;V=a5FNtiB%K&Ny*'
    '8}=5Qd}f&HR^m8e1e@i~0d9PH4dh{)O-B3Zq6p$8puz?`oNkksZzOYE=VLOOCNpDl-HA$-e2+jAq!S$FY)AGc!$WvSX~M8D!|u0@'
    ';g)e;&g5nLj#dh=*6==OC+#=2&aW?Tt>+?zuFfZm{zqzB-I|6HN~-6X*i25|tv@pGm9myrqbqE(vdojw$Yjv42;<sAMH{*5rk{yz'
    'p<Y@p(wh+rIQOci$5s`n-K2nt5YO7tpd!}ZK#!VTIJ)(V6osR4fVE(w8D2eq_QrYyCNncuI%TsPMa_aWi*4I5mh0~^T@j-g@)iOI'
    'sK9WHWI(uuR1wV*gRzSl^|5>}sV()QKc2rIjXyZ`WIm?Ir4v)7=nzU#B$T}q9r3q7GwQ!Gj8a3^jgDI$2bKgr*qFPaYZ2!wQIH(~'
    '?9M0C&z<1_BKHs~f$9EeBDOK;VX9!GSlu_>9~1XpGL2hj?o7PpG{J>{=C43!I$bb;IW=1nXcW197I(3RoV4a5jkzUJQmB*l$zTxQ'
    ')=5o1K=PbUrhP3~^0ln_S!TrvmQWmo?85B96X1>Xa+g1Ury|x{mg_A#RUQYcRYm1BR&}b34_ee#&dX~Iv2cUpQrkSpC(;|YYs|k-'
    'dct0xgiN=6?0Z(_VV-B_wO!K*`JG=^gCs#TmJK<D;PQ7}+f<<-uTa4Y8?QGUe<qgFYz3q7w!=lTdT)!c!bSWt<a;x!Z9%pg>T#v5'
    'V5@Q7zoMhcVK*_RgU{LVG0%DD`LE|KjrgWv6X9jEnW9BBt;@@p<ylYDK#bF9GEXh4FulqUR0e|?!@ekIsMC>x>oyF%H75qHh>Noq'
    'K&#aAtoG#O_4D8TXnYV1g+IHTfafsUlqZ^UTciEl#gsM~I}OB@Uj=@raOhy4B+2edm4<VJZ6cVB0?#b4pI&m%8xuIHz-z4B9r6_g'
    '9ud4K^EDo78v7UVh)baj=E~7!w^C1z824PrZ|1k;8Z?>?>;n}0xrm64#$fM>?N6?2QWiYL#tJq~sb|6xO6Uau6+xKQ6vIq{Eief#'
    '(&BU3OSE28StPoG29}%(oa(=PyRm_nuWc%*O#Rt~G$0(F@t{X@*@B53sl8B&00T|lJ+5h?vNSngF@&IRfzkyFsWji#w4ULWXKtc|'
    'Z%n!JuA+Ytjc5PUq!l)6*S4tb+IO}cP#EsyupxBj2^a#)4=CMWU18CoQ+BX0sm;dgzwmBeJ8EyTuefD_Meh{ia$ki2?U}6}Fl}Hx'
    '{v%6-{(TU>?I2|2Ajb`dw=|MFRcn$ccw7s^dNW#f<+`p)1?)ujw}7YcmA61dk7Cq`=H1~a>W{iZ<h=l=u@I2oaNmAwPQ!v1^q7jx'
    'i4Cu7>r1c8Y?zs-A`XNZkZgW}qD3Z|ado5FNWH|To8)L&*Ubn6JMb$;T7X`+6govm83y+na>uafSCAKBNWy(oG>~y_0N>IJ+wf>R'
    'jMMmvgH7UVpR-~OO(_@OT?iPZDWvz8CksFCL}E7i9@@Sg#k%8bl8M9=%fd@Bmyzi}Vj*@`xDE?YtNs@==2~(&G}w`zqPHju+25W&'
    '_lM|t{b{HD+u7N(chL`p*>t7IudklGK6&{#`e9vi-CTNI%)QB{OADQU!A$O4e=@kslh@}(b6_ZSn~rS4Dh`N;W0fVwK1&^7i$W6t'
    '@Li*(`J9Y7qv_-#T1hIa=XJ!RMlojztpkUOk^LN!RAorh+J)gqgPfdNC$$OBt|W}r_q-Yn13MgTGS+&{;o3TKYJ&LbW56(|Zj*o7'
    '^UQf%tyBPv3e^bARc67TOE9Vzdb&mlt-F6p<lBp-9Sv2jS_<69hN9^a62e3l=o;-uaAZaF<-Nm_@=2mkWS!OYC`uYv2ZA41IaW}O'
    'n9=O;uK(l=i`bWpY=+1%c?lLBvWb<_%Q1^vL~iaAufJkyqzND;`08rT1#U<8GhS;w$K|6QF;e%yUtKXa?0rWI*{N-HM*lO&Mt)+c'
    '$jhfC&2S)<+=W+j!H1ewe!4)Ul_XgTe%!Cs5mbGLu2ULVr<)2|af~rlqWea<O&ng`L7`kncOG)PhYg#tM${5q8Li+oW6aLaQRSNY'
    'z7}`uL!V7-?~fL95m*_gqla!3Vd!f<o1`EkS+!@V)0M@=X~HQ_qxj8>lW4x8rBwfPdCEZBppj52GE%1|t%kr&g>6niO&BdlGn3l7'
    '<U)(3II-Fn7l6_RauG|S7Ei$DwkWMdpw?h`1xMV(z;bCo3JC+LmCxww%eU_P6OX_2Zr<-?n7v!`ZxhKvLRk$uVI!s!$lLoDneVph'
    'ss9xBWoJ5P-+21i@>La=$ykXbyVoKE8wkz#QiF>aY!FtCA5*4nM%`e-l$@)_49M(=Th`o}z(9nJa$VH&|Kn(6GZr<a(54yAZ%nB1'
    '&;}Yu%UM^TSL@NapvBvcdyA_nU`!R-Zys4616I}#=V}o;NBX-czM3F~CUHJfS_pqC;MpR6*Y>3)0{C*G*nz6Wc|f(BU;xi0>-dOq'
    '1Sf-m)=JV0C8SA4APj40pOLt?mcwE4ZUMTVynFVB&Rrc2U+!M1u!GmJf7SeKbW>SaH`jUI{$~x&F8Tz$_i3My$gZ-8PJ_2gVobM)'
    'Awjy?V4F&x^sm{1DaZWAc+Lm`KXcM2j2eGi*6pDE%j4f)uD!?p^To+K@78MXlT_6gBIzF#j>!?^@0L#rvHA4ltJiGp6q?1SwZk`l'
    'ye9XN@Q%30{!ukI_4tR_n#25KY&X=(#s!!=GxegI#6#VU8aB^bLFy!L@Y$2&;B7dmik==g*~_UBj8p)PS71T97<DgGHj)B?cjEHf'
    '>MF;9%~T)4Bt}P#Hay^;nPtjsFoXfsILEkM@L>7Yuu?;|56Dh`_Zsj(SYCDz*EP=YQqF}&%{7>KJ}&PKn*GE;84NvplG++wE!(G8'
    '{oQM=ZbuUj#fp}1$yR!NqLpGh^pdMd|7bT}6Nb%|b6S8;WT|b6Wt`sd66z_l)Ud1Z5#H+kn0hNDoUtEx0%0RSXptSmmY&S1)I(YO'
    'wxxXH&SrMqxGO!TCXF9f!)wjpXE~OJDbah*KdM9AlWgKOH=TJKc6&{tZEq^rpIJkG4C1Sp(!~xNTMGgccl9B&Zl3Hp)-*nakR`Hw'
    '?^dtA-4!<HdCSd-@eH;4qw#6@=+q{{NoN8C<P<e0>@0RJIvFie%NG$9&bg<$k)rjm=xIS~2l*n|)^G;iwmbj%>$8_}UjG*QAw2N('
    'xYO=nkPJ8|tA~}a0nf0|(G9#hAz=&|j(Hss^pwl>S{T8jUVw1<FG!WXH4vvT6bI+xR*||!LOmdd#Q7XMEKD4Db`BYu8fC9rThv#('
    'SYGE_EU%&ARXqv@rN|gw=V7``6S?B@F>A~Qh%l;D#1+g2mu!qJ$yjK=2Hq!fp0%T~>&a{>I^Lc3?=K(4)^W(XkE+UmHb{;rnn*hU'
    'zm&>;@R=XfN6e%_h7)FTFyZt8Uls`oGGa)RPAy;Rvc;PqHUtkUtifd$G|})Va{1rqPu{_h$0GB|Rx7~2*;!%cc6uN*SlFEh?B^On'
    'I7omAv`q4e(OWE{J4RP8Wt;I`ru|mxwv~fiD+e~UjC$i4YAb^(vp<~YS#2jrx_8CV`TEia1=OY;%(=af<wh``$5{wG?3?RhI_!;-'
    'P-V5*Rn1=6;*X{o9iZpO)&lIF36zuz98U|p#Uu-apa04N!%Hp=7SuR*0cV~82Z2)J<c1%5cKD@j?rG|BXkR8R7F=m4zIcep(NK6a'
    'tV=v!PRc;!a56>HV_~6$K-4H$8I#)5iDTxz#w40<trwN>=rffEL*dqlt8{HcWXwo;a-_SL-Qn174C20Dof-a{zC8N%IlldeyP}-f'
    'Xpg!iBd1xu5^A9iTrlHksuIBqFzC%<r-w(P30crI03I3A8O@<fowc8wzXI0!V+6bqGiK1?J?E$SG{<~+Y3u03;CGZVN=<Gw<DmH&'
    'g}Cg`T=uSZt(O$ae7T;!Kl<R3%q;c$dd#&{E>*zZ9x;@nxlPej`oZ<c28{MTP2hWsL=v)7fu{A}TACWM))joij3>d8b7R{E_>L83'
    'G~gotBi=51HA<!2b(7V{DXqN|^~7i*Vl}g~j^Q@`Zk`w#=QuuiW<2!VyLx?Rtwpc=yBc}|ErIV7X61TUH=j?k$N40~hQgGRhmBJ4'
    'Z42t!2=3i9I^hj}%9~sq0dX}hp-FL~3i}(H)6Md!f5tt@G%w2hxtZ@~U%_I5#m*i35!Q{cF|3MYA?a6d#$99^Wdp2{r}Glgu$og%'
    'oU7O~-DjLXr=Q3uWp{69#TwW5zEHcivM)eqMv~@LC85QI$rq#Jy`Q6Gy144lwSkDS^vi4>r|DfzET2qMsKt?i{yiPc4@F8AlSMw8'
    'Kpw;qXI1mV*Q)h~%_1)lw;Wx@_8CA8Xg1;|ot;(mf~$*0f)SaZv)hlJASWbKR_u?(m2~SRq<<K3n)wR5@S1G(AkauFZZCJM;1&U5'
    '*)-G|8IH@6|9C0ceOK6|vm|NJ2nA8x?MO|-(M{8Y=8QHSolf9E(ZsxbD*$ih=m(x3Bo1S#6GJ?b?ievj8!dB1&2`1kbvlf;ScUl+'
    'D~s0#I(YnF9`Zolfy-;ttz|DBBL0vwXyMAyOZmcg5Pi(<&3SnJ;DNbmP?I0{3WFk%YW|t;DCz)JV18l;ReJIdeuABdQSi}*m1<*#'
    '#)hBGQUuH+A19i(CD=Yj_c5-^AYsNFD_D`(e3Ho&Du5r=1ALjx6#iIv7nEu^J4YJ$5SU9%ucSGUb&hk0)}cASy`KyRVV$yuMYs3G'
    '>9<u_dEVu8%Itl3M-a@8#jp2g`L-9Gy>VW?{JH6Q1qH3$iTYE%)xunoa<QofF7azRs8vn+;Vzy26qk+;kJbu)XqrBXaac`srYvDQ'
    'ypPPoRsn>z`RiX)-bH`f=&;LF=Xc*$SdVV5!MX5$IH?O~Y7XX)(M{;p>2YakY7Jge#3UkMPcN{bTV(fiq#V)6vP7d|r;jhZ*umaO'
    'N8Ym28_;(u^hF%ge%>WpEGKr^2SxX6%U`H@Ciryb7=Op5Y^}$+M6F*#FOs66^Glk1jYM)jesuX7Ypw5iS7+e0nniQGzN|e=nwYn{'
    'dmdfrL)jlIoEzL8dw2=9z$*X?^Wnj{!SLcPJ4SW)Z&*Xz-T#BXEdwO{ZHWn3(@vK=@Vlz~=a}_7*Pc>kRp?WAU5+N}6SUACeUDZ6'
    'AWEk7=O;KRf}gi$OZ;DL_8nH5LBWg^5Hw>Ln~ONc6ab_aPxm7CjUj6Vcwzs72;<Yo>@t;9jxN+zlOrUV?(4k%awyj&1Mv#}J!iEj'
    'V@JP_EeO=JJJB_6V?4QWPi|UkY3QBJA%HR<P((K#o@YBC=Kzt`ZLQ919P_BkG8DVPye+5U-h+q(>)3r=(L7l`06Q4kRSM00#0Ruh'
    'WLN%^5xn%S7p2KeOeV^B9vMyxc<RK;MK!REsYyR>q5r7*%|GkHkg6I{*hh^e6a+3GYRhnZKRSMsdYvpTKQi93Tsf@S)ASS21+n`D'
    'jiOGIP;mi{oLVP-PVC}vzLo=}MR>?TVYKgw@w9K|xk~-tvXhn!Fl*=Yhx5<EAM&RxWB6~b?%!P9hFgi3p3#}yjw)HjM3@eaRaV6s'
    'svYjBFNVQde=*6$Gfh=1W*;@8n{J05mPvT-SVq(4grlAP`pLPaIDEW6CsPSQ76R}WZ&n(Tj6KgU$y*&~;tYO2JuA#JMQ<|OdR>_u'
    'dTV~p#*xdV=6g5lpzi3s>u5;9?Ris`oLA<0sVi4<XP4e_h26fH0WzJ|ruH|qbW_?e)cW7h!5=)-gTJAr<=UYNm+W}FMtoDneNF$~'
    'V4nDMjo!7)PL4srI{#Sbf(QKhfbaMXEKtLEnQv#vv+Pf9qlhYCJMS;v&b{D4u~&S=hgXhIx5Xz|zV!Tl><R0#($Vt${P~7I|G}SC'
    '0^rZkPWQNckIVPCe2>fbxO|Vx_qcq|Pv7@-$LxuPlwkt}V9I2ZavS~z)|1#WPo;Gt%ML<|&eJx@MT&*u%hy|lV$7dg1*l@`Xv7~+'
    'FnFl_wsyOAEXR=%Oy^KFW@xP)%u>d(M0nI&jAdDp3gE~wB##|%)$z;Aw?23AkZbKIwt>^$;o|e+PmVPH1QneR&)>V(zF%+)rbB)y'
    'l+-rR0X@`01N2b0Lcs}cG%6=`RdRycjD2(?kLuo|KjhJY_h^}IwPAnknO5(4kF?c8?~%58-+NR*3gmvaL5en^bTnBOx(m33)3+z3'
    'Z`c}*Txe2wy{JIDM($$?d-$Oo+0P<uyZ4zaWyjswIOOYGut{<G#zrYsZj2*Hs*`V*Lzx;D3Zh~<R<tx0ye#DNPR5XyGln#b<W^`4'
    '&Y6O}E=_AH^KmpyoNw0vq7CQOjNLe}imSO323o!+#B?-E=W6*j*6q6%;y)EUl)P2D#ljSc#G*7+zjv?Pm7tphseO$BU>f~{-<;Jm'
    '`qp+O8=gcPrRlZh#0?+5mjB;wqmH;iwmEwyXQybjKv{eGyJu}S8rrp$zk~JRR@SO&8&)sp{s|nc3@yy)^VZb+&KVP;@7d?G+CE3}'
    'm*$0(#_MEnq3nTvi)T2e<C3RBR>4mBPld9A5%ZreixNy`h}@*Qn|6zMnA*mYe@IEA6PA?bN{l9OlHEcGC)JWEe6zi<(E=0~D^}@E'
    'GdjJA4M&<+1AUFhJBQXlDj1jn&o-F>i;RG0IV0dsB;bF0dvDfQfAG1*FIx{l0ng(dhXNu~+IBbsrsw)w3?Ez6%-^704o}^6?!;aH'
    '`?PITpie>{$?w#nhLS+-aCZbB2ywef0{@>KnD19gWb_e@?YYM{a@SG82^MX|u#}T6n7PiLNO{bBcJ^r6pXwj7XVm_5@S)(aW|gxp'
    'iv!6ZBol?Py*mgYlhV$6@+dP$f2aKMZPJJL)_0;oI$!sWyGS7#=Nq9NiQt&fpB8_u$=BM4R_MM3MfW{r`o|N&!<<i$6d;*1j_hhU'
    '7u%A!*|AYWy=+a$X5ykdQ!wMvV6y0q5)I)^1^}!5Dm`?sl62bbC(-0{{}zb{K>iyf(S!M9*8g~yM51JjBry9QkG&sH-X!C@{l%br'
    '_t9^UPaaKZc2;u~xn%&Wc|Oo)hmi&t^&pnz)!#Vs<DtVDrLxWDTCmZq*PX(=awBg3xebmnMhFdmbI1pU<=&Zc`1Y>tu{%1<73WjY'
    'dKMYDplOCC`Q=S?90B|)=t=b&TJ2r!72^#c&vMJ_sONPg)!Ju;UDZ7_4*r$RL8kRrowGMaWty2KdY5DL^E2M6NuM~cgjq9=G3{W<'
    'rh}35pH$CH%|_<A#cMJjlS)^kN?$#GPMa4|ii`0mq20%OfDR5d8rMp<ZZQI`|B(7UY+$pnG07jnWNag1BOd3V*}P8rv^5`SrDJXh'
    'xUFV)VOO5fh))cbYpOYETi%c7$=v-O-$ntRJPW_8F%+dum1?<495LY|7XR6;?vEE&uanEP9GMp_(L=UhwrL4ZaWKN)1W&~Kv0E!@'
    'q_&}vTDFm|ywG;)XNz&{&n6%CW|K~JI+-0uOClTVf3>i>!SiU9eJ~uP8JzC$_xUnggNP|-4dLxDnz+wE;&R`>2FYUc{2QcX%sW7B'
    'aEyGbRcT2ycYn~GcM&@e70`4)Eh<&Zmb&zdiP4;j)q`Tv>1jc82f0p*N;UGGo)&a^FW>2CF%R40n~cgHW}6&OW>-sBHni4Jh5H3>'
    'baLS}9)uxLg`;c_%hOH8e=2H!v7th(n(ts^`C7IIu$#+$5J{p7$5HJ7v<2OcH#Jkwce}BCW{JO=JVCB2o=Mjd&lECU%k56rFVTrE'
    '{L-gGM5A)9G~^<etZG*a$;xxX1T~iH@5Qs1(NCq!efZ|r*N>id9zA{X=$G)l%h3<Ujb@xA>TZH>8qFp>(`4c8%Mrg+N!&>MHqCvb'
    'haYE!Xi_qY8Nnd9jSxVby=$L{Lpy(F{>a|BXV2c9W#Rk_Mr>31Np2$|Ze8Vem{pJ6xODKC`1BjsI^BJ(t@LL4881d7`pGq99ypF-'
    'y3fH9>iR!r(X{wTt};Wj$c457+&oJ3q^RIBM?dDv+yQ)HFvV7~TdG{2Q-ShhKsgEoqIrVPg)j`SpZ|LDIJ;jsa}V`LJ0Kh;Lk5*>'
    'pg4XUA`8=aKD=B^7Icr(mbEX;C!oj9R;LlM8N4QN$#oTe@D04|{vrwNc#6!vsOCvfpMg|{y_N~9;BT}ykL351r(0*x+xKBR{!4qE'
    'd4~g1EDQo>*W)bEE4sq|Hc#li1_m;~?BtY_+pl%wl0j=Pr7Wi`Zb0VM3dZ8^Whq*;b#Ekv08dL;d1J^3e-uDphNK|@hM_u<ZJV1E'
    '&o{dL;gcoq*(D))k_3RUX;Gto?@(c2G?w$Qp_TcCT*$?}L+g-NV%ISE-ep$GZh9FD2JRkdhmLSns4o-gq4@JdiEq+(aSO%H#Sf#L'
    '1|bexD>-85Prt2Sv;vYQgtlacSeT8J;Y-%v4_R7X4{3Q3G0%w{EH1tN^=;dGlPzT2MD+{VmbZ!B7%bgho$v%ofsOZwmOth4tCJ!D'
    'kH?s+A;4GOnYdTWM4N==j4K_t!dM`|%1BUF(GfLZJel24d?%V;>%L6de*Wtlyg<W$NSEf@k}gZWWWk%*5oX`y@rm~)n>I<c@WPLa'
    '2#VMIvzKoeagbd-9hrmXf^({ngf|>Vu8$vrn^kuLFR(ok!NLSj?MoJ&o{4bFnX^MrbN9lahgf_alVhb0Al2s+<hb(}FbE{HdSg=w'
    'n}c;Btoq5QRQylbnK`2`vB|1X$r@8g9(^*KP3UTk5-8D)fZFy1)VNTdE$7AK7v0rkiI8lw$oPU|!67(`Kv_z!jEk_}Z*9kZStR~T'
    '-Ab~mq-xSuB^qC%^N0?N@d~}PC5M}4{qD}2O+0NjFO&JB#cY<0=TKQmDGM;c85N@*mXYodM&+eldflB3yYTOU>Vvl}TtUH?OG2eV'
    '6oV6Cex10d^a>{M39>AIPLe4DnJqX2Ckkuey7rfJuMLKjU|Axs69?B15(zK235Sq*c|o1&R<QaJe>*&99nGW3_+og8ti%`C!GC_$'
    'b-TFSSIYrC9SpAyV!MwyKDgw{vGu_#D0-+j&%GiVPYX>Qs}^upuH4V66{3gpR%U#5@o8C9?}=puY2n;K1tnj33MW~!wy$pTo4D?d'
    '_NSYRpCGMEBH$`^z^pvtt>T^$r908bH5X(IrukU1uAM$Re^K%V{0rkiJXyuJsd}+qs2olGj5)j3H3xTEkSJ%(u!?$r@Zf=s2iHD0'
    '=Tv$HUC;3`T~aR1ZQ&4hYvJpyTetoP^DTQt'
)
MOE_CPP_SRC = r"""
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
"""
_MOE = None
_MOE_B4 = None


_MOE_TINY = None
_MOE_TINY2 = None
_MOE_MID = None
MOE_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")              # the moe agent's MAIN_DEFINES
MOE_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2")    # their small-M build ( () = absent )
MOE_TINY_MAX_M = 4
MOE_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  # their second small-M build ( () = absent )
MOE_TINY2_MAX_M = 8
MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')      # their mid-M build ( () = absent )
MOE_MID_MAX_M = 16


_MA_PRMT_SCALES = True
_MA_SFX_IF_PRMT = ("-DOKC_PRMTSC=1",)


def _ma_scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True


MOE_HAS_PDL = True   # their extension exports okc_set_pdl (programmatic stream serialization)


SC6_WGS = True      # lossless 6-bit block scales in the WGS stages (moe-scale SC6, harness/make_sc6.py)
_MA_SC6W = False    # set in prepare(): scale bytes all in 0x40..0x7F and the PRMT decode and a WGS build
_MA_SFX = ()   # set in prepare() from the weights' block-scale bytes, before any moe build starts


_MA_EXTS = {}
_MA_ALT_SRCS = {}
_MA_MAIN_NS = 3
_MA_PDL = False   # set per call (okc_set_pdl)


def _ma_ext(defines=None):
    defines = tuple(MOE_DEFINES if defines is None else defines)
    if _MA_SC6W and "-DOKC_WGS=1" in defines and "-DOKC_SC6W=1" not in defines and "-DOKC_SC6OFF=1" not in defines:
        defines = defines + ("-DOKC_SC6W=1",)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _MA_EXTS:
        src = MOE_CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        for d in defines:
            if d.startswith("@"):
                src = _MA_ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        import hashlib as _native_build_hash
        source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS))).encode()).hexdigest()[:16]
        _MA_EXTS[defines] = load_inline(
            name=f"glue_moe_{source_tag}_sm{arch}",
            cpp_sources=MOE_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)
    return _MA_EXTS[defines]


def _moe_build(defines):
    """Their own extension builder (lifted verbatim as _ma_ext: flags, DQ7 define, module name), called with the
    build's defines + the weights' suffix, exactly as their kernel_fn calls _ext."""
    ext = _ma_ext(tuple(defines) + tuple(_MA_SFX))
    if MOE_HAS_PDL:
        ext.okc_set_pdl(0)
    return ext


def _moe(M=None):
    """Their kernel, in the build their kernel_fn would pick for this token count."""
    global _MOE, _MOE_TINY, _MOE_TINY2, _MOE_MID, _MOE_B4
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        if _MOE_TINY is None:
            _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
        return _MOE_TINY
    if M is not None and _moe_tiny2_ok() and M <= MOE_TINY2_MAX_M:
        if _MOE_TINY2 is None:
            _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
        return _MOE_TINY2
    if M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        if _MOE_MID is None:
            _MOE_MID = _moe_build(MOE_MID_DEFINES)
        return _MOE_MID
    if (M is not None and 16 < M <= 32 and "_main_defines" in globals()
            and tuple(_main_defines(_MA_SFX)) == tuple(MOE_DEFINES)):  # bet4_claude: decode 32 gets the B4FIN WGS build
        if _MOE_B4 is None:
            _MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))
        return _MOE_B4
    if _MOE is None:   # their M >= 64 rule: _main_defines(sfx) (the WGS build needs the PRMT decode, else fallback)
        _MOE = _moe_build(_main_defines(_MA_SFX) if "_main_defines" in globals() else MOE_DEFINES)
    return _MOE


def _moe_mid():
    global _MOE_MID
    if MOE_MID_DEFINES and _MOE_MID is None:
        _MOE_MID = _moe_build(MOE_MID_DEFINES)
    return _MOE_MID


def _moe_tiny2_ok():
    """Their kernel_fn's rule: the TINY2 build only with the PRMT scale decode when it is a WGS build."""
    return bool(MOE_TINY2_DEFINES) and (bool(_MA_SFX) or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES))


def _moe_tiny2():
    global _MOE_TINY2
    if _moe_tiny2_ok() and _MOE_TINY2 is None:
        _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
    return _MOE_TINY2


def _moe_tiny():
    global _MOE_TINY
    if MOE_TINY_DEFINES and _MOE_TINY is None:
        _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
    return _MOE_TINY


_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


# DQ7: a bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights (the kernel gets -DOKC_DQ7=1)
DQ7 = True


def _stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
                    # register R reads source bit (o + d) mod 32: r0 as stored, r1 = word << 4, r2 = rotl 10,
                    # r3 = word << 6 except the e1 bits (output 8 / 24), which come from rotl 14
                    d = (0, 28, 22, 18 if b == 2 else 26)[R]
                elif R == 0:
                    d = 0
                elif R == 1:
                    d = 6
                elif R == 2:
                    d = 10
                else:
                    d = 4 if b == 0 else 28   # m via rotr 4, (e0, e1, s) via rotl 4 == rotr 28
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos


_POS = _stored_pos()


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)


def _sc6_ok(*sfs):
    """SC6: every scale byte is in 0x40..0x7F (bits 7:6 == 01), so byte - 0x40 is a lossless 6-bit value."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x40) | (b > 0x7F)).any()):
            return False
    return True


def _sc6_unit(s):
    """SC6 skinny unit: [nblk, 4096] scale bytes ([warp 8][lane 32][16 scales]) -> [nblk, 3072]: three planes [warp][lane] of 4 bytes,
    plane w byte n = 4 v_(4w+n) | (v_(12+n) >> 2w) & 3 with v = scale byte - 0x40 (lossless; the kernel's sc_load rebuilds the 4 words)."""
    nblk = s.shape[0]
    v = s.view(nblk, 8, 32, 16).to(torch.int32) - 0x40
    assert int(v.min()) >= 0 and int(v.max()) <= 63
    planes = [torch.stack([(v[..., 4 * w + n] << 2) | ((v[..., 12 + n] >> (2 * w)) & 3) for n in range(4)], -1).to(torch.uint8) for w in range(3)]
    return torch.stack(planes, 1).reshape(nblk, 3072).contiguous()


def _sc6_wgs(s):
    """SC6 WGS piece scales: [..., 1024] bytes (thread t = 8 bytes s0..s7 = [j][h]) -> [..., 768]: plane W (t * 4 B: bytes 4 v_n | piece_n,
    n = 0..3) then plane H (t * 2 B: bytes 4 v_(4+n) | piece_(4+n)); v6 = pieces 0..2, v7 = pieces 3..5 (see harness/make_sc6.py)."""
    lead = s.shape[:-1]
    v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
    assert int(v.min()) >= 0 and int(v.max()) <= 63
    v6, v7 = v[..., 6], v[..., 7]
    piece = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | piece[n] for n in range(4)], -1).to(torch.uint8)
    hb = torch.stack([(v[..., 4 + n] << 2) | piece[4 + n] for n in range(2)], -1).to(torch.uint8)
    return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()



def _fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _kperm_src():
    """KPERM3: stage position p -> stage column k. p = 16 s + c (k16 step s = 4 lg + j, fragment column c of lane
    t = (c >> 1) & 3, i = (c & 1) | ((c >> 3) << 1)); k = 16 (4 t + j) + 4 lg + i, so lane t's columns of step j in all
    four load groups are exactly scale block 4 t + j."""
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (4 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(256))
    return k


def _kperm_codes(codes):
    """codes uint8 [R, K/2] (low nibble first) -> the same codes with K permuted inside every 256-K stage."""
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def _kperm_rows(rowsA):
    """Gate/up row order whose 16-row block r16, lane g holds inter column colmap(P), P = 8 r16 + g, where
    colmap(P) = 256 (P >> 8) + _kperm_src()[P & 255] is the column phase B reads at K position P (the phase-A
    epilogue then stores at its usual position)."""
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 8) * 256 + _kperm_src().to(rowsA.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


def _wgs_copy(codes, sf, rows, sc6=False):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, t, j (block 16 st + 4 t + j)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, warp, g, t, j, h (KPERM3)
    if sc6:
        s = _sc6_wgs(s)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


# the M >= 64 build when the scale bytes rule out the PRMT decode (the WGS kernel needs it): the mma.sync kernel
MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def _main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(MOE_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MOE_DEFINES


def _wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (MOE_DEFINES, MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES))


# ---------------------------------------------------------------------------
# Dense projections: the muse_bf16_proj entry's kernels (every name bfp_*),
# used where the tuner finds them faster than cuBLAS or the Triton split-K.
# The CUDA template strings and their generators below are kept verbatim.
# ---------------------------------------------------------------------------
bfp__SK2_DELAY = int(os.environ.get("SK2_DELAY_ESC", "0"))  # debug only: delay (ns) of escape warp 0 + ring spin-limit trap


bfp_CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
// cross-CTA spins are bounded (POD.md Oct 26 17:50) in the measured-free form (attention v298, POD.md 18:31): count
// polls, no timer read per poll, and past 2^24 polls call an out-of-line trap (keeps the hot path's codegen unchanged)
__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }
__device__ __forceinline__ void okc_spin_ge(const int* p, int target, int ns) {
  for (unsigned it = 0;; ++it) {
    if (*((volatile const int*)p) - target >= 0) return;
    if (ns > 0) __nanosleep(ns);
    if (it >= (1u << 24)) okc_spin_trap();
  }
}
@@BODY@@
"""

bfp_MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v86y = v86r + bf16_proj v131's hints): weight streams are read once per call -> evict_first; x tiles
// are re-read by every CTA -> evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef()) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}

"""

bfp_WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


def bfp_wgmma_defs(bns):
    """Generate wgmma_call overloads for m64nBNk16 bf16 with A from registers."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


bfp_TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
// on sm_80 all 32 producer lanes issue 16-byte cp.async and arrive with cp.async.mbarrier.arrive).
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  // dg (diagnostics only, 0 in production): 1 = skip epilogue, 2 = also skip x / escape-table loads,
  // 3 = also skip all compute (pure weight streaming through the pipeline)
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;   // keep the loads
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;   // (the producer warp has already returned)
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + rb, target, 20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

bfp_MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el()) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

bfp_GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)   // A tile pre-swizzled and contiguous: [rb][kb][BM x 128 B]
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    okc_spin_ge(counters + rb, target, 32);
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


bfp_SK2_SRC = r"""

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
// PDL (programmatic dependent launch) for the sk2 launches: 0 = off (plain launches); N > 0 = launch with programmatic
// stream serialization and issue the first N weight stages before griddepcontrol.wait (set by sk2_set_pdl)
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
#define SK2_DIAG 0
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  // FUSE (pooler + router in one launch): after its pooler stages every CTA streams one router piece: 8 router rows
  // (tile rt = cta / 4) over one quarter of K (rq = cta % 4), from the router's own stage-major hi|lo stream
  // ([16 stages][32 tiles][8 warps][2048 B]); 4 more stages through the same ring
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
  // PCX: pdl bits 0-3 = P (pre-wait stages), 4-8 = d0 and 9-13 = g (pacing, units of 448 cycles)
  const long long pcx_d0 = (long long)((pdl >> 4) & 31) * 448, pcx_g = (long long)((pdl >> 9) & 31) * 448;
  const int pcx_e = (pdl >> 14) & 7;
  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA's own stages)
  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)   // PDL: the first P stages' weights go out before the wait; their barriers first
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // weights never depend on the previous kernel: under PDL (pdl > 0) they stream while it finishes; x (its output)
    // only after griddepcontrol.wait (a no-op when the launch has no programmatic dependency)
    const long long pcx_t0 = clock64();
    for (int st = 0; st < P; ++st) {
      const long long pcx_tg = st < pcx_e ? 0 : pcx_d0 + (long long)(st - pcx_e) * pcx_g;   // PCX pacing: stage st goes out pcx_tg cycles after entry
      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if constexpr (CL == 1) {
      // PCX hoist: every barrier init and the escape-table copy go out BEFORE the wait (they need nothing from the
      // previous kernel); after the wait thread 0 only issues the x tiles, the producer lane the refills
      if (COMP) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      if (!(dgv & 4))
        for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      if (COMP) {
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL == 1) {   // PCX hoist: the x tiles of the pre-issued stages, right after the wait
    if (tid == 0 && !(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
  }
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);   // every escape warp hands every stage back
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
      if (dgv & 8) {   // diagnostic: replace the tensor-core instructions by ALU work
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue
  if constexpr (FUSE) {
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
    // k-warp reduction of the 8 x TOKS piece through the (now idle) ring: redr[wk][tok][8]
    if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warps are done with the ring too
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    // this CTA's quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)
    // adds them in quarter order (deterministic) and writes the router output
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {   // every CTA: SM id, start and end time
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


bfp_WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
  };
  const bool producer = tid == 0;
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // h = 0: hi tile, 1: lo tile (same B) -> FP32 sum of both
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }  // !HLW
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    const int old = atomicAdd(counters + zh * nrb + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + zh * nrb + rb, target, 32);   // bounded (POD.md Oct 26 17:50)
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  __threadfence();
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def bfp_wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


bfp_CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);
void sk2_set_pdl(int64_t n);
"""

bfp_w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
bfp_COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}

bfp__ext = None


bfp_TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
bfp_USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
bfp_WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def bfp__tma_nstage(nw, comp, sm90, ks, hl=False):
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = (112 if sm90 else 40) * 1024
    return max(2, min(8, budget // (nw * chunk)))


bfp_TMA_TU = {}  # (layer, mtk) -> 0 / 1 (which translation unit instantiates it)


# alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
# against the default config and cuBLAS on every scored shape
bfp_TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def bfp__tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(bfp_TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(bfp_TMA_ALT.items())]
    return items


def bfp__tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in bfp__tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], bfp__mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = bfp__tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else bfp_w_rows[layer]
        wgm = bool((sm90 or bfp_WGM_SM80_TEST) and comp and bfp_USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    keys = list(lines)
    for ln in keys:
        for cfg in lines[ln]:
            bfp_TMA_TU[cfg] = 0
    return bfp_TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


bfp_GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


bfp_DIAG_LAYERS = ()
bfp_GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256


def bfp_swizzle_tiles(w, bm):
    """W [N, K] -> [N/bm][K/64][bm rows x 128 B] with the 128B swizzle (16-byte chunk c of row r at c ^ (r % 8))."""
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rb, kb, r, c, 8]
    r = torch.arange(bm, device=w.device).view(bm, 1)
    p = torch.arange(8, device=w.device).view(1, 8)
    src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
    return torch.gather(t, 3, src_c).contiguous()


def bfp__gemm_src(sm90):
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(bfp_GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and bfp_GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in bfp_GEMM_CONFIGS})
    return (bfp_GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", bfp_wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


bfp_GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
    // stage the accumulators in this CTA's (now idle) ring, then one bulk shared::cluster copy into rank 0's buffer
    float* mine = reinterpret_cast<float*>(gbase);   // [NWG][NACC][128]
    if constexpr (NWG > 1) named_bar(1, NWG * 128);   // every warpgroup is done reading the ring
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

bfp_GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 7),
}
bfp_USE_GQ = True


def bfp__gq_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(bfp_GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return bfp_GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def bfp__run_gq(x, layer, m, n, k, state):
    nwg, cl, ns = bfp_GQ_CONFIGS[(layer, bfp__gq_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    bfp__get_ext().gq(state["raw"][layer], x, out, m, n, k, bfp__gq_bn(m), nwg, cl, ns)
    return out


bfp_WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    # router: BF16 hi|lo tiles (plain), 64-row blocks
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def bfp__wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def bfp__wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(bfp_WIDE_CONFIGS.items()):
        hl = layer == "router"
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return bfp_WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__get_ext():
    """Compile everything as one CUDA translation unit with load_inline.  no_implicit_headers: the CUDA file includes
    only the ATen headers it needs (torch/types.h alone costs ~45 s of nvcc time)."""
    global bfp__ext
    if bfp__ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS if sm90 else "")
        defs = bfp_wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + bfp__tma_srcs(sm90) + bfp_MAP_SRC + bfp__gemm_src(sm90) + bfp__wide_src(sm90) + bfp__sk2_src(sm90)
                    + bfp__gq_src(sm90))
        bfp__ext = load_inline(
            name=f"dl_okc_bf16proj_v86y4_{arch}" + (f"_dly{bfp__SK2_DELAY}" if bfp__SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + bfp_CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={bfp__SK2_DELAY}"] if bfp__SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return bfp__ext


def bfp_pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def bfp_pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    n_esc = int(esc.sum())
    if n_esc > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    # fragment order for low bytes and codes: [rg, s, lane, 8]
    lo_f = bfp_pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = bfp_pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * 0x01010101,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def bfp_build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def bfp_build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


# ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
bfp_SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv16_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler16h_r100', 'unpooler', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
]
bfp_SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o8h_r88',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8_r176',
    ('qkv_gate', 16): 'qkv16_r176',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16h_r100',
    ('unpooler', 32): 'unpooler32_r144',
}
bfp_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
bfp_USE_SK2 = True
bfp_WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def bfp__sk2_name(layer, m):
    return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]


def bfp__sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def bfp__sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(bfp_SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap = int(tail[1])
    ring = min(cap * 1024, 232448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def bfp__sk2_src(sm90):
    lines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if bfp_SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    flines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if name != "pooler":
            continue
        for tokr in (8, 16, 32):
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
            flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                          f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                          f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                          f"rcnt.data_ptr<int>()); return; }}")
    return (bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))
            .replace("@@SK2F_DISPATCH@@", "\n  ".join(flines)))


def bfp_sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def bfp_sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = bfp_sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def bfp_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


def bfp_sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, delta bits)."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["delta"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.view(torch.int32)
    return tab.contiguous(), pmax


def bfp_sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (bfp_USE_SK2 and state["sm90"]):
        return
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = bfp_sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            comp = state.setdefault("sk2_comp", {}).get(layer)
            if comp is None:
                comp = bfp_sk2_compress(w.contiguous())
                state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                shared[lkey] = (bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW),) + tuple(bfp_sk2_table(comp, n_rows, R))
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)


def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = bfp__sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def bfp_fused_pooler_router(x, state):
    """Pooler and router of one decode step in ONE launch (m <= 32, sm_90): x [m, 8192] bf16 ->
    (pooler [m, 4096] bf16, router [m, 256] fp32). Same numerics as the separate sk2 kernels: 12-bit lossless pooler
    weights with exact escape corrections, router as BF16 hi + lo with FP32 accumulation (quarter-K partials added in
    a fixed order)."""
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m))
    return out, outr


bfp_SK2_PDL = 0   # decode_layer: set to N > 0 for programmatic dependent launches (N weight stages before the wait)


def bfp__mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def bfp_prepare(weights):
    ext = bfp__get_ext()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
    old_paths = not (sm90_ and bfp_USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    # split-K partials: kernels run one at a time, so every layer shares one workspace
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * bfp_w_rows[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * bfp_w_rows[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in bfp_GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    if bfp_GEMM_DIAG:
        state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256)
            # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
            src = torch.stack([bfp_pack_fragments(hi), bfp_pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = bfp_WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
            for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            c = bfp_pack_compressed(w) if name in bfp_COMPRESS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = bfp_pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
            key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
            ckey = (name, splits, rgc, wsn, ks, n_cfg)
            if ckey not in state["cnt"]:
                state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
            if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
                key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
                if key not in state["streams"]:
                    state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
                ckey = (name, splits, rgc, wsn, ks, n_cfg)
                if ckey not in state["cnt"]:
                    state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
                if (name, splits, rgc, n_cfg) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
            if lname != name or not comp:
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = bfp_build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["old_paths"] = old_paths
    bfp_sk2_prepare(weights, state)
    if bfp_GEMM_DIAG and state["sm90"]:
        state["wtile"], state["diag_cnt"] = {}, {}
        for layer in bfp_DIAG_LAYERS:
            state["wtile"][layer] = bfp_swizzle_tiles(state["raw"][layer], 128)
            for a_s in (0, 1):
                for sp_ in sorted({bfp_GEMM_CONFIGS[(layer, 128)][1], bfp_GEMM_CONFIGS[(layer, 256)][1], 1}):
                    state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(bfp_w_rows[layer] // 16, dtype=torch.int32, device=dev)
    bfp__selftest(weights, state)
    bfp__autotune(weights, state)
    torch.cuda.empty_cache()
    return state


# Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
# "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
bfp_ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
bfp_WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
bfp_ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def bfp__route(layer, m, sm90):
    for mmax, r in (bfp_ROUTE_ROUTER if layer == "router" else bfp_ROUTE_BF16):
        if m <= mmax:
            if r == "gemm" and not sm90:
                return "torch"
            if r == "wide" and not (sm90 or bfp_WIDE_SM80_TEST):
                return "torch"
            return r
    return "torch"


def bfp__route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, bfp__sk2_name(layer, m), bfp__sk2_tokr(m))
    if route == "skinny":
        return (layer, route, bfp__mt_key(m))
    if route == "wide":
        return (layer, route) + bfp__wide_key(m)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def bfp__selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(1234)
    for layer, w in weights.items():
        n, k = w.shape
        for m in bfp_SELFTEST_M:
            nn = 1024 if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in bfp__candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else bfp__route_key(layer, m, ch[0]))
                try:
                    out = bfp__run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError:  # launch failure -> fall back
                    err = float("inf")
                if not err < tol:
                    state["disabled"].add(key)


def bfp__run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = bfp__mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = bfp__get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, bfp_w_rows[layer] // 16, ks, dg)
    else:
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, bfp_w_rows[layer] // 16, ks, dg)
    return out


def bfp__run_wide(x, layer, m, n, k, state):
    tok, nz = bfp__wide_key(m)
    nwg, splits, _ = bfp_WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    bfp__get_ext().wide(state["streams"][(layer, 0, rgc, 1, 4, n)], x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def bfp__run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = bfp_GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    bfp__get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def bfp__run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def bfp__run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind == "skinny":
        return bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return bfp__run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return bfp__run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return bfp__run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return bfp__run_gq(x, layer, m, n, k, state)
    return bfp__run_torch(x, layer, m, n, k, state)


def bfp__static_choice(layer, m, state):
    route = bfp__route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (bfp__sk2_name(layer, m) in state.get("sk2", {})
                              and bfp__route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not bfp_WIDE_BF16:
            route = "torch"
    if route != "torch" and bfp__route_key(layer, m, route) in state.get("disabled", ()):
        # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and bfp__route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = bfp__mt_key(m)
        return ("skinny", bfp_TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def bfp__candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [bfp__static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in bfp_TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", bfp_TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and bfp__route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and bfp__sk2_name(layer, m) in state.get("sk2", {})
            and bfp__route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (bfp_USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, bfp__gq_bn(m)) in bfp_GQ_CONFIGS
            and bfp__route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


bfp_AUTOTUNE = True
bfp_TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def bfp__autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not bfp_AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)   # 256 MB, read-only sweep
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    reps = 24
    for m in bfp_TUNE_M:
        for layer, w in weights.items():
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = bfp__candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                bfp__run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    bfp__run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()


def bfp__nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "gq" and (layer, bfp__gq_bn(m)) not in bfp_GQ_CONFIGS:
        return None
    return ch


def bfp_kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or bfp__nearby_choice(layer, m, state) or bfp__static_choice(layer, m, state)
    out = bfp__run(choice, x, layer, m, n, k, state)
    if bfp_GEMM_DIAG:
        bfp__diag(x, layer, m, n, k, state, choice)
    return out


def bfp__diag(x, layer, m, n, k, state, choice):
    """Extra kernels for the judge's per-kernel profile (outputs discarded)."""
    if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
        for dg in (1, 2, 3):   # same kernel without epilogue / x loads / compute
            bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)


def _bfp_prepare(weights, st):
    """Their prepare() on the five dense matrices, guarded: any failure just disables the backend."""
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
    except Exception as ex:  # noqa: BLE001
        print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


BFP_PDL_N = 2   # bf16_proj's sk2 / fused kernels: weight stages issued before griddepcontrol.wait (0 = off)


PCX_CFG = {}   # (role, T or 0) -> (P, d0, g[, e]); see harness/make_pcx.py


def _pcx_val(role, T):
    P, d0, g, e = (tuple(PCX_CFG.get((role, T), PCX_CFG.get((role, 0), (BFP_PDL_N, 0, 0, 0)))) + (0,))[:4]
    return (1 << 30) | P | (d0 << 4) | (g << 9) | (e << 14)


def _bfp_pdl(T, role=None):
    """Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue."""
    global bfp_SK2_PDL
    bfp_SK2_PDL = _pcx_val(role, T) if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m, name)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


_MA_MMAX = 384
_MA_EMAX = _MA_MMAX * 8


WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16   # moe WGS_FIXF: stream-K fixups + doneAs after ybuf (zero between calls)
WGS_XROWS = _MA_EMAX + 8 * (32 + _MA_EMAX // 32 + 8)  # >= moe WGS_XROWS for FT_T >= 32


def _df_inter(inter):
    """DF: inter rows [128, 256) start (and, after every MID launch, end) as bf16 0xFFFF = not yet written."""
    inter[128 * 4096:256 * 4096].view(torch.int16).fill_(-1)
    return inter


def _ma_make_scratch(dev):
    return [torch.zeros(8 + 1024 + _MA_EMAX + 4096, dtype=torch.int32, device=dev),  # counters + doneA + ss copy 1 + MFIN flags
            torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
            _df_inter(torch.zeros((_MA_EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev))]  # inter, then WGS xA / xB

SCRATCH_GUARD = 1 << 16   # sentinel words after each buffer my glue allocates: wide enough for far overruns (v154m-style state ~4K words out)
SCRATCH_SENTINEL = 0x5EADBEEF


def _ma_prepare(weights, st):
    dev = weights["w13"].device
    I = H = 4096
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(32):
        wA.append(_pack_codes(weights["w13"][e].view(torch.uint8), rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e].view(torch.uint8), rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    st["ma_w"] = [torch.cat(wA), torch.cat(sA), weights["w13_gs"].float().contiguous(),
                  torch.cat(wB), torch.cat(sB), weights["w2_gs"].float().contiguous(),
                  lut.contiguous(), weights["post_expert_norm"].contiguous(),  # bf16; kernel forms 1 + w
                  torch.cat([_fat_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_fat_order(x, y) for x, y in zip(wB, sB)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wB, sB)])]
    if "_wgs_on" in globals() and _wgs_on():   # their prepare(): w[12], w[13] = the WGS copies (per 256-row unit)
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA), _MA_SC6W)
                                     for e in range(32)]))
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, _MA_SC6W)
                                     for e in range(32)]))
    # Scratch is sized for every build kernel_fn may pick: a scratch sized
    # from the 64-token formula once let the 32-token MID build write past it.
    st["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(dev)]


def _to_i32(x64):
    x64 = x64 & 0xFFFFFFFF
    return (x64 - (x64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def _repack_q(wp, interleave):
    """wp uint8 [E, R, 2048] (low nibble = even k) -> int32 [E, R/16, 64, 32, 4] mma-fragment words."""
    E, R, _ = wp.shape
    if interleave:
        wp = wp.view(E, 2, R // 32, 16, 2048).transpose(1, 2).reshape(E, R, 2048)
    x = wp.view(E, R // 16, 16, 256, 8).permute(0, 1, 3, 2, 4).to(torch.int64)  # [E, RB, kb, row, j]
    B3 = x[..., :8, :4]
    B2 = x[..., 8:, :4]
    B1 = x[..., :8, 4:]
    B0 = x[..., 8:, 4:]
    w = ((B0 & 15) | ((B1 & 15) << 4) | ((B2 & 15) << 8) | ((B3 & 15) << 12)
         | ((B0 >> 4) << 16) | ((B1 >> 4) << 20) | ((B2 >> 4) << 24) | ((B3 >> 4) << 28))
    w = _to_i32(w).reshape(E, R // 16, 64, 4, 32).permute(0, 1, 2, 4, 3).contiguous()
    return w


def _repack_s(sf, interleave):
    """sf uint8 [E, R, 256] -> uint8 [E, R/16, 64, 8, 8]: (g, [rows g kb0..3, rows g+8 kb0..3])."""
    E, R, _ = sf.shape
    if interleave:
        sf = sf.view(E, 2, R // 32, 16, 256).transpose(1, 2).reshape(E, R, 256)
    x = sf.view(E, R // 16, 16, 64, 4).permute(0, 1, 3, 2, 4)  # [E, RB, ch, row, kk]
    return torch.cat([x[:, :, :, :8, :], x[:, :, :, 8:, :]], dim=-1).contiguous()


# ----------------------------------------------------------------------------------------------
# dense skinny GEMM: P[s, t, n] = sum_{k in split s} X[t, k] * W[n, k]
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # X is the producer's output
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Like _gemm_splitk with x = rmsnorm_per_64(X) * sigmoid(G) applied on the fly (X bf16, G fp32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr, mask=tmask, other=0.0)
        x3 = tl.reshape(x, [BK // 64, 64, BT])
        r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
        xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
        acc = tl.dot(w, xn.to(tl.bfloat16), acc)
        w_ptr += BK
        x_ptr += BK
        g_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
    tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
    N, K = w.shape
    if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
        xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
        _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        return dense(xn, w, T, state, "o")
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    S = 1
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
    """The tuned token count whose choices apply to T (the next one at or above it)."""
    for t in TUNED_T:
        if T <= t:
            return t
    return TUNED_T[-1]


BFP_MAX_T = 32   # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those
                 # loads); at T=256 their routes measured 10 us slower than cuBLAS on 32x8 sliding, at T=128 a wash


def dense(x, w, T, state=None, name=None):
    """Returns (partials [S, T, N] fp32, S). The tuner picks cuBLAS or the split-K Triton kernel per
    (token count, weight matrix): cuBLAS wins on the o projection, the Triton one on the pooler+router."""
    N, K = w.shape
    pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
    if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
        # bf16_proj's kernel (BF16 out, as SGLang's GEMM): the consumers read P as one [1, T, N] partial
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick is True or pick == "bfp":       # above BFP_MAX_T a bfp pick runs as cuBLAS (measured in the chain)
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    if T > 32:
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    tiles = (N // BN) * triton.cdiv(T, BT)
    S = 1
    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
        S *= 2
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@triton.jit
def _f32_to_e4m3(x):
    """float32 -> e4m3fn bits (int32), round to nearest even, saturating to +-448."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    # subnormal: multiples of 2^-9
    sub = (a * 512.0 + 0.5).to(tl.int32)  # round half up (ties are measure-zero)
    ab = a.to(tl.int32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    r = r & 0x7FF00000
    e = (r >> 23) - 120
    m = (r >> 20) & 7
    nrm = (e << 3) | m
    nrm = tl.minimum(nrm, 0x7E)
    code = tl.where(a < 0.015625, sub, nrm)
    return code | sign


# ----------------------------------------------------------------------------------------------
# QK norm + RoPE + cache write
# ----------------------------------------------------------------------------------------------
@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    base = P + t * 2304
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    d = tl.arange(0, 64)[None, :]
    qe = tl.zeros([16, 32], dtype=tl.float32)
    qo = tl.zeros([16, 32], dtype=tl.float32)
    ke = tl.zeros([2, 32], dtype=tl.float32)
    ko = tl.zeros([2, 32], dtype=tl.float32)
    v = tl.zeros([2, 64], dtype=tl.float32)
    g = tl.zeros([1024], dtype=tl.float32)
    gi = tl.arange(0, 1024)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for s in range(S):
        b = base + s * T * 2304
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + hk * 64 + d)
        g += tl.load(b + 1280 + gi)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        ang = pos * tl.load(INVF + i)
        c = tl.cos(ang)
        sn = tl.sin(ang)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = QOUT + t * 1024 + hq * 64 + 2 * i
    tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
    tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
    q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
    tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
    tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
    tl.store(GOUT + t * 1024 + gi, g)
    loc = tl.load(LOC + t).to(tl.int64)
    kinv = 1.0 / tl.load(KS)
    vinv = 1.0 / tl.load(VS)
    kb = KC + loc * 128 + hk * 64 + 2 * i
    tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
    tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
    tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


# ----------------------------------------------------------------------------------------------
# attention (split-KV flash decoding over a paged FP8 cache)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
    """One program = one request, both KV heads (128-byte K/V rows), one key split."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(SEQ + b)
    m_off = tl.arange(0, BM)
    l_idx = m_off // 8
    hh = m_off % 8
    row_ok = m_off < L * 8
    qpos = seq + l_idx
    d = tl.arange(0, 64)
    qrow = (b * L + l_idx) * 16 + hh
    q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    total = seq + L
    if SLIDING:
        lo = tl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    n = total - lo
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
    kst = lo + s * per
    ken = tl.minimum(total, kst + per)
    scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
    m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
    m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BM], dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    a0 = tl.zeros([BM, 64], dtype=tl.float32)
    a1 = tl.zeros([BM, 64], dtype=tl.float32)
    for kb in range(kst, ken, BN):
        offs = kb + tl.arange(0, BN)
        kv_ok = offs < ken
        page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
        row = page.to(tl.int64) * page_size + offs % page_size
        kptr = row[:, None] * 128 + d[None, :]
        ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
        if SLIDING:
            ok = ok & (offs[None, :] > qpos[:, None] - window)
        k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        # head 0
        sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
        mn = tl.maximum(m0, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m0 - mu)
        l0 = l0 * al + tl.sum(p, 1)
        a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
        m0 = mn
        # head 1
        sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
        mn = tl.maximum(m1, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m1 - mu)
        l1 = l1 * al + tl.sum(p, 1)
        a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
        m1 = mn
    ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
    ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
    tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
    tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
    tl.store(ML + ob0 * 2, m0)
    tl.store(ML + ob0 * 2 + 1, l0)
    tl.store(ML + ob1 * 2, m1)
    tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
    """One program per (token, q head): merge the splits, per-head RMSNorm, * sigmoid(gate)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // L
    l = t % L
    g = h // 8
    m_row = l * 8 + h % 8
    d = tl.arange(0, 64)
    base = (b * 2 + g) * NSPLIT
    M = tl.full([SB], float("-inf"), dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
    Mx = tl.max(M, 0)
    Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
    den = tl.zeros([SB], dtype=tl.float32)
    num = tl.zeros([64], dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        rows = (base + sidx) * BM + m_row
        m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
        l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(m_s - Mx)
        den += w * l_s
        a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
        num += tl.sum(w[:, None] * a, 0)
    o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
    if NORMGATE:
        o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
        gt = tl.load(GATE + t * 1024 + h * 64 + d)
        o = o * tl.sigmoid(gt)
    tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# boundaries, routing
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
    """o projection without a K split (K = 1024), so each program owns complete rows: it writes a and
    accumulates the four per-token sums the post-attention boundary needs (a^2, (a1 s)^2, (a1 s)(b1 a), (b1 a)^2)."""
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    x_ptr = X + rt[None, :] * K + rk[:, None]
    for _k in range(0, K, BK):
        w = tl.load(w_ptr)
        if NORMGATE:
            xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
            gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
            # per-64 rmsnorm * sigmoid(gate) on the fly (BK is a multiple of 64)
            xr = tl.reshape(xv, [BK // 64, 64, BT])
            rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
            xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
            x = xv.to(tl.bfloat16)
        else:
            x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
    st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
    a1v, b1v = _gate_ab(A1, rn)
    as_ = a1v[:, None] * st
    ba = b1v[:, None] * acc
    ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
    q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
    q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
    q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
    q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
    m = rt < T
    tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    """s1 = a1 * stream + b1 * a * rstd(a), h = s1 * rstd(s1) * g, from the four sums (one block per column slice)."""
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a = tl.load(AOUT + t * N + offs)
    a1v, b1v = _gate_ab(A1, offs)
    s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
    tl.store(S1OUT + t * N + offs, s1)
    tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
    """Reduce the pooler/router partials: one block per (token, pooled slice); slice 0 also does the top-8."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    c = j * BC + tl.arange(0, BC)
    pooled = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        pooled += tl.load(P + (s * T + t) * 4608 + c)
    tl.store(POOL + t * 4096 + c, pooled)
    tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
    if j == 0:
        e = tl.arange(0, 256)
        logit = tl.zeros([256], dtype=tl.float32)
        for s in range(S):
            b = P + (s * T + t) * 4608
            logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        ids = tl.zeros([8], dtype=tl.int32)
        wv = tl.zeros([8], dtype=tl.float32)
        for k in tl.static_range(8):
            idx = tl.argmax(sel, 0)
            val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
            ids = tl.where(k8 == k, idx, ids)
            wv = tl.where(k8 == k, val, wv)
            sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
    """x = rmsnorm(pooled) * (1 + pre_expert_norm), in the expert kernel's layout."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
    x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
    if PERMUTED:
        ch = c // 64
        kk = c % 64
        kb = kk // 16
        rr = kk % 16
        po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
    """f = (unpooler partials summed) * g_postffn, and the four sums for the FFN boundary."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < NU
    f = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
    f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
    tl.store(FOUT + t * 8192 + c, f)
    st = tl.load(S1 + t * 8192 + c)
    a2v, b2v = _gate_ab(A2, c)
    as_ = a2v * st
    bf = b2v * f
    tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
    tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
    tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
    tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a2v, b2v = _gate_ab(A2, c)
    s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
    tl.store(SOUT + t * N + c, s2)
    tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
    """alpha, beta of a residual gate, from the checkpoint's BF16 weight (as SGLang's fused module does)."""
    w = tl.load(G + offs).to(tl.float32) / 0.3
    beta = tl.sigmoid(w)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha, beta


@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, ROUTER_OUT, WRITE_ROUTER: tl.constexpr = False, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    a1v, b1v = _gate_ab(A1, offs)              # weights and the layer's input stream: no producer dependency
    stv = tl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the o-projection) is the producer's output
    a = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        a += tl.load(P + (s * T + t) * 8192 + offs)
    n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n
    tl.store(S1OUT + t * 8192 + offs, s1)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))
    if WRITE_ROUTER:
        tl.store(ROUTER_OUT + t * 8192 + offs, hh)


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    pooled = tl.zeros([4096], dtype=tl.float32)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (pooler + router) is the producer's output
    for s in range(S):
        b = P + (s * T + t) * 4608
        logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        pooled += tl.load(b + c)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    # BF16 activation, stored as exact fp16 in the MoE kernel's fragment-permuted layout
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Same as _route but the pooler output (bf16) and the router logits (fp32) arrive as separate tensors."""
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # pooled / logits are the producer's output
    logit = tl.load(LOGITS + t * 256 + e)
    pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# MoE experts (NVFP4 W4A16)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _pexp_norm_sum(x_ptr, w_ptr, s_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps,
                   BLOCK_H: tl.constexpr):
    """post-expert norm (weight first) * routing weight, summed over slots. Same kernel as SGLang's
    fused_post_expert_norm_weighted_sum (Apache 2.0), called directly: its wrapper falls back to a chain of
    ~8 torch ops below 32 tokens, which costs 39-43 us on the judge instead of ~6."""
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)   # the checkpoint's BF16 w; 1 + w formed here
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    m = offs < NU
    gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)      # weights, and S1 from _boundary1: every kernel between them (pooler/router, route,
    gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)      # moe, unpooler) triggers only after its own wait, so _boundary1 completed before this kernel began
    a2v, b2v = _gate_ab(A2, offs)
    s1v = tl.load(S1 + t * 8192 + offs)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the unpooler) is the producer's output
    f = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
    f = f * gp
    n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n
    tl.store(SOUT + t * 8192 + offs, s2)
    hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------

def _marlin_bank(weights):
    """The local experts in Marlin's layout (SGLang's own loader step), as baseline.py builds them."""
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous().clone(),
        w2_weight=weights["w2"].contiguous().clone(),
        w13_weight_scale=weights["w13_sf"].contiguous().clone(),
        w2_weight_scale=weights["w2_sf"].contiguous().clone(),
        w13_weight_scale_2=weights["w13_gs"].contiguous().clone(),
        w2_weight_scale_2=weights["w2_gs"].contiguous().clone(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _marlin_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs + post-expert-norm finalize (as baseline.py).
    moe_block_size and use_atomic_add come from the prepare()-time sweep in _tune."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids  # already (expert - local_start) % 256 mapped to 0..31 / -1 by the router kernel
    mc = state.get("marlin_cfg", {})
    cfgm = mc.get(num_tokens, mc.get(_bucket(num_tokens)))
    if cfgm is None:
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        atomic = False
    else:
        block_size_m, atomic = cfgm
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    m = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _pexp_norm_sum[(num_tokens,)](y, state["g_postexp"], topk_weights, m, local_ids, TOPK=TOP_K,
                                  H=size_k, eps=1e-8, BLOCK_H=4096, num_warps=8, num_stages=2)
    return m


def _graph_time(fn, flush, reps=9):
    """Median graph-replay time with a cold L2. The flush is a READ sweep: zeroing a buffer leaves ~120 MB of
    dirty lines to write back, which inflates the next kernel by 15-25% (measured on the judge's H200) and not
    evenly across kernels, so it would skew every choice below."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        _FLUSH_SINK.copy_(flush.sum())
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
    """Give the 32 local experts a skewed bias so that synthetic routing matches the judge's busiest-range load:
    ~1.6 local routes per token and ~15.7 distinct local experts at 32 tokens (reference statistics). Two
    parameters (skew width, bias size) found by a torch simulation of the real router; no input data involved."""
    router = st["pr"][D_EXPERT:].float()  # [256 hi; 256 lo] rows of width HIDDEN
    router = router[:N_EXPERTS] + router[N_EXPERTS:]
    gc = torch.Generator(device=dev).manual_seed(987654321)
    h = torch.randn(512, HIDDEN, generator=gc, device=dev)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    sc = torch.sigmoid(h.double() @ router.double().T * ROUTER_MULT).float()
    idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

    def stats(shape, delta):
        sel = sc + st["bias"]
        sel[:, :LOCAL_EXPERTS] += delta * shape
        ids = sel.topk(TOP_K, dim=-1).indices
        loc = ids < LOCAL_EXPERTS
        rpt = float(loc.float().sum(-1).mean())
        counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
        d32 = sum(counts) / len(counts)
        return rpt, d32

    best = None
    for width in (2.0, 3.0, 4.5, 6.0, 9.0):
        shape = torch.exp(-idx / width)
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if stats(shape, mid)[0] < 1.6:
                lo = mid
            else:
                hi = mid
        rpt, d32 = stats(shape, hi)
        if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
            best = (shape, hi, rpt, d32)
    st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
    if log:
        print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
    """prepare()-time choice of the post-attention path per token count: the persistent megakernel or the
    multi-kernel pipeline, and which routed-expert kernel. Uses synthetic activations only (never input data)."""
    dev = st["qkv"].device
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)  # 240 MB, read-swept between replays
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(12345)
    # Synthetic routing is uniform over all 256 experts, but the judge picks each input's local range as the
    # busiest one: ~1.6 local routes per token (2.6 at T=1) over few distinct experts. Calibrate a skewed bias
    # on the local experts (torch-side simulation with the real router weights) so the tuned load is realistic.
    bias0 = st["bias"].clone()
    _calib_bias(st, dev, g, log)
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
                "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        if T <= 4:  # redraw until the token(s) route to about as many local experts as the judge's inputs do
            want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
            for _ in range(40):
                _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
                    break
                data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
                o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        best, bestt, bestv2 = None, 1e30, True
        times, dfltv2 = {}, False
        msg = []
        nloc = ndis = -1
        for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                           ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            if st.get("bfp") is not None:
                st["mm_pick"][(T, name)] = "bfp"
                try:
                    if name == "pr":
                        t_bfp = _graph_time(lambda: (_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                     _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                            flush, reps=5)
                    else:
                        nn = w.shape[0]
                        t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
                except Exception:  # noqa: BLE001
                    t_bfp = 1e30
            else:
                t_bfp = 1e30
            st["mm_pick"][(T, name)] = False
            try:
                t_tri = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  # noqa: BLE001
                t_tri = 1e30
            st["mm_pick"][(T, name)] = True
            try:
                t_mm = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  # noqa: BLE001
                t_mm = 1e30
            t_bfpf = 1e30
            if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
                try:
                    t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"]), flush, reps=5)
                except Exception:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    t_bfpf = 1e30
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf}
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                msg.append(f"{name}{ {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F'}[bestp] }:{opts[bestp]:.0f}")
        if st["bank"] is not None and T <= 384:
            xm = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            raw = torch.randint(0, N_EXPERTS, (T, 8), generator=g, device=dev, dtype=torch.int32)
            # the convention the router kernel writes: local index 0..31, -1 for another GPU's expert
            mids = torch.where(raw < LOCAL_EXPERTS, raw, torch.full_like(raw, -1)).contiguous()
            mwt = torch.full((T, 8), 0.125, device=dev)
            bestm, bestmt = None, 1e30
            for bs in (8, 16, 32, 48, 64):
                for atomic in (False, True):
                    st["marlin_cfg"][T] = (bs, atomic)
                    try:
                        tm = _graph_time(lambda: _marlin_experts(xm, mwt, mids, st), flush, reps=5)
                    except Exception:  # noqa: BLE001
                        continue
                    if tm < bestmt:
                        bestm, bestmt = (bs, atomic), tm
            if bestm is not None:
                st["marlin_cfg"][T] = bestm
                if log:
                    msg.append(f"mcfg{bestm[0]}{'a' if bestm[1] else ''}:{bestmt:.0f}")
            else:
                st["marlin_cfg"].pop(T, None)
        # my own MoE kernel (FP16 expert input) is offered only where neither SGLang's Marlin nor the moe agent's
        # kernel is available (never on the judge image): both beat it everywhere, and SGLang keeps BF16 there
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
                 and (c != "agent" or st["ma_ok"])]
        for moe in cands:
            if moe == "marlin" and st["bank"] is None:
                continue
            if moe == "agent" and not st["ma_ok"]:
                continue
            st["moe_pick"][T] = moe
            for v2 in (False,):  # Parallel boundary search lost every complete-chain sweep.
                st["gluev2"][T] = v2
                try:
                    t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    if log:
                        msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
                    continue
                if log:
                    msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
                    if nloc < 0:
                        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                        loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
                        nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
                if t < times.get(moe, 1e30):
                    times[moe] = t
                    if moe == DEFAULT_MOE:
                        dfltv2 = v2
                if t < bestt:
                    best, bestt, bestv2 = moe, t, v2
        if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
                and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
            best, bestv2 = DEFAULT_MOE, dfltv2
        st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
        st["gluev2"][T] = bestv2
        if log:
            print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg), flush=True)
    st["bias"].copy_(bias0)
    # The megakernel comparison used to live here; it never won, so pbmax is
    # always 0 unless FORCE_PBMAX overrides it. The state key is kept.
    st["pbmax"] = 0 if FORCE_PBMAX is None else FORCE_PBMAX
    if log:
        print("tune: pbmax", st["pbmax"], "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
    del flush


def prepare(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st = {}
    st["router_fp32"] = router.contiguous()
    st["qkv"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
    st["unpool"] = weights["unpooler"].contiguous()
    st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
    # Every BF16 checkpoint weight stays BF16. Norms apply (1 + w), formed in
    # FP32 at use; the post-expert weight is stored as SGLang stores it.
    st["gate1"] = weights["attn_residual_gate"].contiguous()
    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["g_postexp"] = weights["post_expert_norm"].contiguous()   # BF16 as given; _pexp_norm_sum forms 1 + w
    st["g_postffn"] = weights["post_ffn_norm"].contiguous()
    st["g_next"] = weights["next_pre_attn_norm"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    st["w13q"] = _repack_q(weights["w13"].view(torch.uint8), True)
    st["w13s"] = _repack_s(weights["w13_sf"].view(torch.uint8), True)
    st["gs13"] = weights["w13_gs"].float().contiguous()
    st["w2q"] = _repack_q(weights["w2"].view(torch.uint8), False)
    st["w2s"] = _repack_s(weights["w2_sf"].view(torch.uint8), False)
    st["gs2"] = weights["w2_gs"].float().contiguous()
    st["invf0"] = torch.zeros(32, device=dev)
    st["invf"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))).contiguous()
    errs = []

    def _build(fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            errs.append(ex)

    global _MA_SFX, _MA_SC6W
    _MA_SFX = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    _MA_SC6W = bool(SC6_WGS and _MA_SFX and _wgs_on() and _sc6_ok(weights["w13_sf"], weights["w2_sf"]))
    print(f"[sc6] scales ok -> wgs sc6 {_MA_SC6W}", flush=True)
    st["ma_sfx"] = tuple(_MA_SFX)
    st["ma_ext"] = {}
    ths = [threading.Thread(target=_build, args=(f,)) for f in (_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid, bfp__get_ext, _graft_glue_module, _graft_b2_module, _sl_cluster_module, lambda: _moe(32))]  # ~85 s of nvcc, in parallel
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ma_prepare(weights, st)
    if _MA_SC6W:
        try:
            for d in (tuple(MOE_DEFINES), tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)):
                print("[sc6] regs/local/smem (<false>, <true>)", d[-1], _ma_ext(d + tuple(_MA_SFX)).okc_attrs(), flush=True)
        except Exception as ex:
            print(f"[sc6] attrs failed: {ex!r}"[:200], flush=True)
    _bfp_prepare(weights, st)
    st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
    st["fa3"] = None
    st["bank"] = None
    st["att_ok"] = True
    st["ma_ok"] = True
    st["pbmax"] = 2
    st["moe_pick"] = {}
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        st["fa3"] = flash_attn_with_kvcache
        st["bank"] = None  # Native experts cover supported shapes; omit unused Marlin packing.
    except Exception as ex:  # noqa: BLE001  (local GPUs without sgl_kernel: fall back to our own kernels)
        print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  # noqa: BLE001
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    for k, v in PIN_PICK.items():      # in-chain winners the isolated tuner sometimes misses (no PDL in its timing)
        st.setdefault("mm_pick", {})[k] = v
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     # every attention plan family once, on synthetic inputs
    _check_guards()            # after the tuner (every moe build) and the attention self-test
    return st


def _attn_cfg(B, L, sliding):
    BM = 16 if L == 1 else 64
    if sliding:
        nsplit = max(1, min(32, 264 // B))
    else:
        nsplit = max(1, min(132, 528 // B))
    return BM, nsplit


_INV_FREQ_CACHE = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    t = _INV_FREQ_CACHE.get(key)
    if t is None:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        else:
            t = torch.zeros(32, device=device)
        _INV_FREQ_CACHE[key] = t.float().contiguous()
    return t


def _launch7(data, dbg=None, _mod=None, pdl=None, **override):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = _plan7(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = _TUNED7.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = max(1, min(plan["ncl"], 4))
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    # bit 0: TMA loads (compiled out by default); bit 1: cooperative finalize (all CTAs resident)
    # cooperative finalize: only for the long single-request grids (their last-CTA finalize reads and zeroes
    # tens of KB from one SM); the short sliding grids finalize faster in one CTA (measured on H200)
    if flat:   # grid-wide finalize after all segments (the kernel re-checks residency)
        coop = int(plan.get("coop", int(OKC_FLATCOOP and L == 8 and
                                        n_splits * pairs <= 2 * _num_sms7(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * _num_sms7(device))))
    w0 = int(plan.get("w0", (OKC_W0_L1 if L == 1 else OKC_W0_L8) if (flat and (coop or L == 1)) else
                      (OKC_W0_PR if (coop and not flat and B * pairs == 1) else 1024)))
    use_pdl = OKC_PDL if pdl is None else bool(pdl)
    tma = int(plan.get("tma", 0)) | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (int(use_pdl) << 16)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = _workspace7(B * pairs * nc * L * 16, device)
    n_ctas = n_splits * (1 if flat else B) * pairs
    scratch, bar = _cluster_bufs7(n_ctas if cs > 1 else 1, L, device)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    (_mod or _att7()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, scratch, bar, tma, *(_PF7 or (0, 0)))
    return out


def _supported7(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in (1, 8) and cfg["batch"] <= 512 and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)

ATT_HAS_PDL = True   # does the embedded attention launcher take pdl= (their v63+)?


def _decode_attention7(data, state, qkv):
    """The attention agent's own launcher, lifted verbatim as _launch7: plan, workspace sizes and the coop / w0
    bits always match the kernel version embedded here (v82: my hand copy had drifted -- w0 = 0, no flat coop).
    This layer's config already carries their keys (heads, head_dim, page_size, window, rope_theta, scales)."""
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    global _PF7
    bst = state.get("bfp")
    _PF7 = None
    if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):
        _ps = bst["sk2"]["pooler"][1]
        _PF7 = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
    out = _launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
    _PF7 = None   # the pointer belongs to this call's state only
    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)


# Guards around every component buffer this glue allocates or caches (POD.md item 6 follow-up): the moe
# scratch, the attention workspace and its cluster buffers. Each is a view into a larger allocation whose
# tail is SCRATCH_GUARD sentinel words; prepare() checks every tail after the tuner and an attention self-test
# have run each path, so an out-of-bounds write fails loudly there instead of corrupting a neighbour.
_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
    nbytes = t.numel() * t.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
    full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
    v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
    _GUARDS.append((full, words))
    _GUARDED_IDS.add(id(v))
    return v


def _check_guards():
    torch.cuda.synchronize()
    for full, words in _GUARDS:
        bad = int((full[words:] != SCRATCH_SENTINEL).sum())
        if bad:
            raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


def _attn_selftest(st):
    """Run the attention kernel once per plan family on synthetic inputs (problem constants, no data), so its
    lazily allocated buffers exist and are guarded before any timed call, and the guards see every path."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4242)
    nq = st["qkv"].shape[0]
    for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                                 (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                                 (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                                 (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
        cfg = dict(batch=B, q_len=L, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=P, window=window,
                   rope_theta=ROPE_THETA if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
        if not _supported7(cfg):
            continue
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        per = (ctx + L + P - 1) // P
        pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
        pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
        loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
        kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.zeros_like(kc)
        qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                            "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
    """The attention entry's launcher host-checks these sizes (32-bit index math, OKC_OFF32) and raises otherwise;
    mirrored here so such shapes take SGLang's FA3 instead (fuzz: batch 64 x 150k context exceeds the caches' 4 GB)."""
    kc, vc = data["k_cache"], data["v_cache"]
    rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
    return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
            and data["page_table"].numel() < (1 << 31)
            and cfg["batch"] * cfg["q_len"] * QKV_ROWS < (1 << 31)
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
    st2 = dict(state)
    st2["att_ok"] = False
    return kernel_fn(data, st2)


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    # q_len 1 only: on the kv_scale = 1/64 test their q_len 8 path's query scaling rounds coarser than FA3
    # (conformance test 9: hidden/stream 1.41x SGLang's error), so FA3 keeps every q_len 8 shape.
    # q_len 8: their kernel on full layers, and on sliding only for a single request, where fusing QK-norm, RoPE,
    # the cache write and the norm-gate into one launch beats FA3 + my two glue kernels by 5.6% in-layer;
    # at B >= 16 sliding FA3 still matches or wins (16x8 -1.2%, 32x8 +0.8%), below my 4% bar for q_len 8.
    agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
    if agent_attn:
        # fused decode attention (QK-norm, RoPE, cache write, split-KV, merge, rmsnorm * sigmoid(gate))
        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
        else:
            Pq, Sq = dense(data["attn_input"], state["qkv"], T)
            qkv = Pq[0] if Sq == 1 else Pq.sum(0)
            qkv = qkv.to(torch.bfloat16)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            # their launcher's host-side size checks run before any launch: nothing half-captured, take FA3
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    dev = data["attn_input"].device
    x = data["attn_input"]
    sliding = cfg["window"] > 0
    use_fa3 = state["fa3"] is not None
    # 1. qkv projection, QK-norm, RoPE, FP8 cache write
    P, S = dense(x, state["qkv"], T)
    q = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
    q8 = torch.empty((T, Q_DIM), device=dev, dtype=torch.uint8)
    gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                   data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                   ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 2. attention
    if use_fa3:
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                           v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
        window = cfg["window"]
        out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                           v_cache=data["v_cache"], page_table=data["page_table"],
                           cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                           max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                           window_size=(window - 1, 0) if window else (-1, -1),
                           scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        o_in = out.reshape(T, 1024)
    else:
        BM, NS = _attn_cfg(B, L, sliding)
        acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
        ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
        pt = data["page_table"]
        _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                             pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                             L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
        o_in = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
        _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                               SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
    return _after_attention(data, state, o_in, gate, T)


def _pick_expert_backend(pick, marlin_available, agent_available):
    """The expert-backend fallback, factored out of _after_attention: Marlin
    without its bank falls back to the agent kernel, and vice versa. The
    three steps are kept in order (not a loop): when neither backend is
    available the original code terminates with "agent", and so does this."""
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    if pick != "marlin" and not agent_available:
        pick = "marlin"
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    return pick


def _after_attention(data, state, o_in, gate, T):
    """o projection onwards: residual boundary, router, experts, unpooler, final boundary.
    gate is None when o_in already carries rmsnorm(o) * sigmoid(gate). state["gluev2"] picks between the
    one-program-per-token boundary kernels and the parallel (atomic-sum) ones; the tuner chooses per T."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gv = state.get("gluev2", {})
    v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if v2:
        # o projection with no K split (each program owns whole rows) + the boundary's four per-token sums
        a_out = torch.empty((T, HIDDEN), **f32)
        sums = torch.zeros((T, 4), **f32)
        BN, BT, BK = 64, (16 if T <= 16 else 32), 128
        _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
            o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
            K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
        BC1 = 2048
        _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                        s1, h, N=HIDDEN, BC=BC1, num_warps=8)
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                         WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if T <= 8 and v2:
        _router_input_norm[(T,)](s1, state["g_preffn"], hr, num_warps=8)
    # 4. pooler + router (BF16 hi + lo rows, FP32 accumulation), routing, pre-expert norm
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
        pooled, logits = P
        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    elif v2:
        pool = torch.empty((T, D_EXPERT), **f32)
        psum = torch.zeros((T,), **f32)
        BCR = 1024
        _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                       BC=BCR, num_warps=8)
        _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                       PERMUTED=not (use_marlin or use_ma), num_warps=8)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 5. experts
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)   # the build their kernel_fn picks for this M, then at most MMAX tokens per launch, as theirs
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):      # (their scratch is sized for MMAX: T > 384 would overrun it)
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])     # lids: local expert index, -1 = another GPU's expert
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    # 6. unpooler + post-FFN boundary + next layer's norm
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if v2:
        fbuf = torch.zeros((T, HIDDEN), **f32)
        sums2 = torch.zeros((T, 4), **f32)
        BC2 = 1024
        _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                       BC=BC2, num_warps=8)
        _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                        N=HIDDEN, BC=BC2, num_warps=8)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids


# ==============================================================================================
# A_graft_glue_d1_v1_v32 graft block (appended to the vjeux base, everything above is unchanged).
# Portions from vjeux/okc-2026 (Apache-2.0): the base above, and _graft_after_attention below, which is
# a copy of the base's _after_attention (non-v2 path) with exact-shape graft gates.
# Grafted parts are our team's own code (Apache-2.0 submission): glue_cluster_boundary (8-CTA cluster
# boundary), _boundary_kernel (Gluon row boundary), the (token, 2) one-warp route-finish structure of
# _pool_route_finish, and the cuBLASLt picks (lead_kq256 qkv m256, dl_lt_kdl_finish m128/m256).
# Gates are exact token counts: T=1 (decode 1), T=8 (verify 1), T=128 (verify 16), T=256 (verify 32).
# ==============================================================================================
import ctypes as _C
import hashlib as _hashlib
from functools import lru_cache as _lru_cache
from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch
from triton.experimental import gluon as _glx
from triton.experimental.gluon import language as _gl

_GRAFT_GLUE_T = (1, 8, 16, 32, 128, 256)      # token counts that take the grafted boundary + route kernels
_GRAFT_LT_T = (128, 256)          # token counts that take the cuBLASLt picks
_GRAFT_ROUTE_EARLY = False   # route kernel: trigger dependents before (True) or after (False) its wait

_GRAFT_GLUE_CUDA = r'''
// A3: the base Gluon lane mapping and floating-point operation order, across
// eight 128-thread CTAs. This source is embedded in the single-file candidate.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
};

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  // start's div.full.f32 by 8192 + eps becomes this FFMA in ptxas SASS.
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}

template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
  // Gluon contracts each even square into its adjacent odd square, then uses
  // the balanced in-thread tree. No sequential sum or reassociation is allowed.
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* p=cluster.map_shared_rank(sm,dst);
      p[rank*warps_per_cta+warp]=sum;
    }
  }
  cluster.sync();
  // Each warp independently reads the original 32 (or 16) warp partials and
  // performs the identical descending XOR butterfly. Separate r1/r2 buffers
  // prevent a fast CTA from overwriting a partial still read by a slow CTA.
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
}

template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N];
  __shared__ float partials[2][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
          // Preserve the signed zero from the post weight as well as the
          // sliced boundary's alpha*stream FMA contraction.
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
        if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
      }
    }
  }
  // DSMEM destinations must exist before the first remote store. All work
  // above this barrier and the dependency wait is independent of the branch.
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
      if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
    }
  }
  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}

template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// graft: boundary 1 (attention residual), BF16 [rows, 8192] branch, rows <= 16, optional FP32 router copy.
void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0)};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
// graft: boundary 2 (FFN residual), full-width BF16 [rows, 8192] unpooler output, rows <= 16 (our kind 1).
void b2c(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)next.data_ptr(),(int)stream.size(0)};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b2c: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,true,false,false,true>,s);
  else glue_launch(a,glue_cluster_boundary<4,32,true,false,false,false>,s);
}
'''
_GRAFT_B2_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
__device__ __forceinline__ void lf2(const float* p, float* x) {
  float2 v = *reinterpret_cast<const float2*>(p); x[0]=v.x; x[1]=v.y;
}
__device__ __forceinline__ void lb2(const BF* p, float* x) {
  unsigned u = *reinterpret_cast<const unsigned*>(p);
  x[0]=__uint_as_float(u<<16); x[1]=__uint_as_float(u&0xffff0000u);
}
__device__ __forceinline__ void glf2(const float* p, float* x) {
  asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
}
__device__ __forceinline__ void gsf2(float* p, const float* x) {
  asm volatile("st.global.v2.f32 [%0],{%1,%2};"::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
}
__device__ __forceinline__ void gsb2(BF* p, const float* x) {
  unsigned u;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u):"f"(x[1]),"f"(x[0]));
  asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u):"memory");
}
__device__ __forceinline__ float pair_sq(float x0, float x1) { return gf(x0,x0,gm(x1,x1)); }
__device__ __forceinline__ float butterfly(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}

// P physical threads; J=1024/P logical threads each; logical L=t+P*j owns columns 2L+2048r+{0,1}, r=0..3.
// Columns < 1024 are exactly r==0 of logical threads L<512, i.e. j<JL (JL=512/P).
template<int P,bool SBF>
__global__ __launch_bounds__(P) void b2_single(const float* part, const float* stream, const BF* post,
    const BF* pre, const float* alpha, const float* beta, BF* out, float* next, int rows) {
  constexpr int J=1024/P, JL=512/P, WP=P/32;
  const int t=threadIdx.x, lane=t&31, w=t>>5, row=blockIdx.x;
  __shared__ float sm1[32], sm2[32];
  const float* srow=stream+(size_t)row*8192;
  float* nrow=next+(size_t)row*8192;
  float ns[J][8], pr[J][8];
  float s0[JL][2], al[JL][2], be[JL][2], po[JL][2];
  float q1[JL], q23[JL];
  // ---- before the wait: everything that does not depend on the expert branch ----
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      const int col=2*L+2048*r;
      lb2(pre+col,pr[j]+2*r);
      if(j<JL && r==0) {
        lf2(srow+col,s0[j]); lf2(alpha+col,al[j]); lf2(beta+col,be[j]); lb2(post+col,po[j]);
      } else {
        float s[2],a[2],b[2],p[2];
        lf2(srow+col,s); lf2(alpha+col,a); lf2(beta+col,b); lb2(post+col,p);
        #pragma unroll
        for(int k=0;k<2;k++) ns[j][2*r+k]=gf(a[k],s[k],gm(b[k],gm(0.f,p[k])));
      }
    }
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) if(!(j<JL && r==0)) gsf2(nrow+2*L+2048*r,ns[j]+2*r);
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    if(j<JL) {
      q1[j]=pair_sq(ns[j][2],ns[j][3]);
      q23[j]=ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7]));
    } else {
      // Logical warps 16..31: whole r2 partial now; their r1 partial is +0 (all-zero branch).
      float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),pair_sq(ns[j][2],ns[j][3])),
                 ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7])));
      v=butterfly(v);
      if(lane==0) { sm2[w+WP*j]=v; sm1[w+WP*j]=0.f; }
    }
  }
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  // ---- after the wait: columns < 1024 only ----
  float b[JL][2];
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    if constexpr(SBF) {
      // graft: the base's sliced unpooler output, one BF16 [rows, 1024] tensor (already BF16-rounded)
      unsigned u;
      asm volatile("ld.global.b32 %0,[%1];":"=r"(u):"l"(reinterpret_cast<const BF*>(part)+(size_t)row*1024+col):"memory");
      b[j][0]=gb(gm(__uint_as_float(u<<16),po[j][0]));
      b[j][1]=gb(gm(__uint_as_float(u&0xffff0000u),po[j][1]));
    } else {
    float pp[8][2];
    #pragma unroll
    for(int sk=0;sk<8;sk++) glf2(part+((size_t)(sk*rows+row))*1024+col,pp[sk]);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float x=0.f;
      #pragma unroll
      for(int sk=0;sk<8;sk++) x=ga(x,pp[sk][k]);
      x=gb(x);
      b[j][k]=gb(gm(x,po[j][k]));
    }
    }
    // A3 local_squares over (b0,b1,0,...,0): ((q0+q1)+(q2+q3)) with q1=q2=q3=+0.
    const float z=pair_sq(0.f,0.f);
    float v=ga(ga(pair_sq(b[j][0],b[j][1]),z),ga(z,z));
    v=butterfly(v);
    if(lane==0) sm1[w+WP*j]=v;
  }
  __syncthreads();
  const float r1=gr(butterfly(sm1[lane]),1.e-8f);
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float bn=gb(gm(b[j][k],r1));
      ns[j][k]=gf(al[j][k],s0[j][k],gm(be[j][k],bn));
    }
    gsf2(nrow+col,ns[j]);
    float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),q1[j]),q23[j]);
    v=butterfly(v);
    if(lane==0) sm2[w+WP*j]=v;
  }
  __syncthreads();
  const float r2=gr(butterfly(sm2[lane]),1.e-5f);
  BF* orow=out+(size_t)row*8192;
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      float h[2];
      #pragma unroll
      for(int k=0;k<2;k++) h[k]=gm(gm(ns[j][2*r+k],r2),pr[j][2*r+k]);
      gsb2(orow+2*L+2048*r,h);
    }
  }
}


// graft: boundary 2 (FFN residual) for the base's T <= 8 sliced unpooler (BF16 [rows, 1024]), one CTA per row.
void b2s(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  int rows=(int)stream.size(0);
  if(stream.size(1)!=8192 || branch.size(0)!=rows || branch.size(1)!=1024) throw std::runtime_error("graft b2s: shape");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(rows);cfg.blockDim=dim3(512);cfg.stream=s;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=1;
  auto e=cudaLaunchKernelEx(&cfg,b2_single<512,true>,(const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
      (const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),(const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
      (BF*)out.data_ptr(),(float*)next.data_ptr(),rows);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''


@_lru_cache(None)
def _graft_glue_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='graft_glue_' + _hashlib.md5(_GRAFT_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_GRAFT_GLUE_CUDA, functions=['b1', 'b2c'],
                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@_lru_cache(None)
def _graft_b2_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='graft_b2_' + _hashlib.md5(_GRAFT_B2_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_GRAFT_B2_CUDA, functions=['b2s'],
                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@_glx.jit
def _graft_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                        H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    # our _boundary_kernel (r20 Gluon row boundary), router copy removed
    _gdc_wait()
    _gdc_launch()
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        w = _gl.load(PostW + c).to(_gl.float32)
        b = (b * w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    s = _gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    w = _gl.load(PreW + c).to(_gl.float32)
    h = ns * r * w
    _gl.store(Out + row * H + c, h)


@_glx.jit
def _graft_row_boundary_pre(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                            H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    # our _boundary_preload_kernel (r20, 1 < m <= 32): weights before the wait, router copy removed
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    h = ns * r * pw
    _gl.store(Out + row * H + c, h)


@triton.jit
def _graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, PACKED: tl.constexpr):
    # our _pool_route_finish structure: program (t, 0) = pre-expert norm of the pooled row, (t, 1) = top-8 pick;
    # one warp each, both after the wait. Arithmetic is the base's _route/_route_split (fp32 pooled, 1 + w in fp32).
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        if PACKED != 0:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if PACKED == 1:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


# ---- cuBLASLt picks (our lima Lt wrapper, CUDA 13 ABI of the judge image) ----
_GP, _GI, _GSZ = _C.c_void_p, _C.c_int, _C.c_size_t


class _GraftAlgo(_C.Structure):
    _fields_ = [("data", _C.c_uint64 * 8)]


class _GraftHeur(_C.Structure):
    _fields_ = [("algo", _GraftAlgo), ("workspaceSize", _C.c_size_t), ("state", _C.c_int),
                ("wavesCount", _C.c_float), ("reserved", _C.c_int * 4)]


class _GraftLt:
    def __init__(self, transpose_a):
        self.lib = _C.CDLL("libcublasLt.so.13")
        sigs = {
            "cublasLtCreate": [_C.POINTER(_GP)],
            "cublasLtMatmulDescCreate": [_C.POINTER(_GP), _GI, _GI],
            "cublasLtMatmulDescSetAttribute": [_GP, _GI, _GP, _GSZ],
            "cublasLtMatrixLayoutSetAttribute": [_GP, _GI, _GP, _GSZ],
            "cublasLtMatrixLayoutCreate": [_C.POINTER(_GP), _GI, _C.c_uint64, _C.c_uint64, _C.c_int64],
            "cublasLtMatmul": [_GP] * 12 + [_C.POINTER(_GraftAlgo), _GP, _GSZ, _GP],
            "cublasLtMatmulAlgoInit": [_GP, _GI, _GI, _GI, _GI, _GI, _GI, _GI, _C.POINTER(_GraftAlgo)],
            "cublasLtMatmulAlgoConfigSetAttribute": [_C.POINTER(_GraftAlgo), _GI, _GP, _GSZ],
            "cublasLtMatmulAlgoCheck": [_GP, _GP, _GP, _GP, _GP, _GP, _C.POINTER(_GraftAlgo), _C.POINTER(_GraftHeur)],
        }
        for name, types_ in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types_, _GI
        self.handle = _GP()
        self.call("cublasLtCreate", _C.byref(self.handle))
        self.desc = _GP()
        self.call("cublasLtMatmulDescCreate", _C.byref(self.desc), 68, 0)
        op = _GI(1 if transpose_a else 0)
        self.call("cublasLtMatmulDescSetAttribute", self.desc, 3, _C.byref(op), _C.sizeof(op))
        self.workspace = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
        self.alpha, self.beta = _C.c_float(1), _C.c_float(0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} status {code}")

    def __deepcopy__(self, memo):
        import copy
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        # Native handles/descriptors are immutable after prepare; workspace belongs to each copy.
        result.lib, result.handle, result.desc = self.lib, self.handle, self.desc
        result.alpha, result.beta = self.alpha, self.beta
        result.workspace = copy.deepcopy(self.workspace, memo)
        return result


class _GraftPlan(dict):
    def __deepcopy__(self, memo):
        import copy
        result = type(self)()
        memo[id(self)] = result
        for key, value in self.items():
            # Matrix-layout handles contain only shape/stride metadata, never tensor addresses.
            result[key] = value if key == "layouts" else copy.deepcopy(value, memo)
        return result



def _graft_layouts(lt, m, n, k, mode, batch, fp32):
    # dl_lt_kdl_finish _ltf_layouts: column-major W @ X.T; batches split output features.
    wa = (k, n // batch, k, k * n // batch) if mode == "tn" else (n // batch, k, n, n // batch)
    result = []
    for index, (rows, cols, ld, stride) in enumerate((wa, (k, m, k, 0), (n // batch, m, n, n // batch))):
        desc = _GP()
        lt.call("cublasLtMatrixLayoutCreate", _C.byref(desc), 0 if fp32 and index == 2 else 14, rows, cols, ld)
        count, offset = _GI(batch), _C.c_int64(stride)
        lt.call("cublasLtMatrixLayoutSetAttribute", desc, 5, _C.byref(count), _C.sizeof(count))
        lt.call("cublasLtMatrixLayoutSetAttribute", desc, 6, _C.byref(offset), _C.sizeof(offset))
        result.append(desc)
    return result


# (base matrix name, T) -> (mode, batch, fp32 out, algo words); picks of dl_lt_kdl_finish (c5e0be33) on our layouts.
_GRAFT_LT_PICKS = {
    ("pr", 256): ('tn', 2, True, [785979015234, 4294967331, 0, 562949953421312, 558779540176897, 60129542158, 292057776128, 0]),
    ("pr", 128): ('tn', 1, True, [575525617730, 4294967331, 0, 1970324836974598, 558779540176897, 60129542158, 292057776128, 0]),
    ("o", 256): ('nn', 2, False, [85899345986, 4294967331, 0, 1407374883553286, 558779540176897, 60129542158, 292058693646, 0]),
    ("un", 256): ('nn', 2, False, [85899345986, 4294967331, 0, 1407374883553287, 558779540176897, 60129542158, 292058693646, 0]),
    # un m128: our head's quebec LtNN algo (batch 2); the ltf pick ran 1.5 us slower in this pipeline (gpu03 CUPTI)
    ("un", 128): ('nn', 2, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
    # our head d0e53792: qkv m128 = lima "qkv_medium" (TN, batch 16); o m128 = quebec LtNN batch 8 (lt_nn_algo)
    ("qkv", 128): ('tn', 16, False, [395136991298, 4294967331, 0, 1407374883553282, 558779540176897, 60129542158, 292058693646, 0]),
    ("o", 128): ('nn', 8, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
}
# lead_kq256 (our head d0e53792 _QKV256_CONFIG): qkv m256, algo 66 built from attributes.
_GRAFT_QKV256 = {"id": 66, "tile": 124, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": 0, "cluster": 2}
_GRAFT_QKV256_ATTRS = (("tile", 1, _C.c_uint32), ("splitk", 2, _C.c_int32), ("red", 3, _C.c_uint32),
                       ("swz", 4, _C.c_uint32), ("custom", 5, _C.c_uint32), ("stages", 6, _C.c_uint32),
                       ("cluster", 8, _C.c_uint16))


def _graft_lt_plan(st):
    lt_tn, lt_nn = _GraftLt(True), _GraftLt(False)
    plans, status = {}, {}
    src = {"pr": st["pr"], "o": st["o"], "un": st["unpool"], "qkv": st["qkv"]}
    for (name, m), (mode, batch, fp32, words) in _GRAFT_LT_PICKS.items():
        if m not in _GRAFT_LT_T:
            continue
        lt = lt_tn if mode == "tn" else lt_nn
        if mode == "tn":
            weight = src[name]
            n, k = weight.shape
        else:
            key = "graft_t_" + name
            if key not in st:
                st[key] = src[name].t().contiguous()
            weight = st[key]
            k, n = weight.shape
        assert weight.is_contiguous() and weight.data_ptr() % 128 == 0, (name, weight.data_ptr() % 128)
        assert lt.workspace.data_ptr() % 256 == 0 and n % batch == 0 and (n // batch * 2) % 16 == 0
        layouts = _graft_layouts(lt, m, n, k, mode, batch, fp32)
        algo = _GraftAlgo((_C.c_uint64 * 8)(*words))
        heur = _GraftHeur()
        code = lt.lib.cublasLtMatmulAlgoCheck(lt.handle, lt.desc, *layouts, layouts[2], _C.byref(algo), _C.byref(heur))
        status[(name, m)] = (code, heur.workspaceSize)
        if code or heur.workspaceSize > lt.workspace.numel():
            continue
        plans[(name, m)] = _GraftPlan(lt=lt, layouts=layouts, algo=algo, weight=weight, n=n, fp32=fp32)
    if 256 in _GRAFT_LT_T:
        algo = _GraftAlgo()
        lt_tn.call("cublasLtMatmulAlgoInit", lt_tn.handle, 68, 0, 14, 14, 14, 14, _GRAFT_QKV256["id"], _C.byref(algo))
        for name, attr, ctype in _GRAFT_QKV256_ATTRS:
            value = ctype(_GRAFT_QKV256[name])
            lt_tn.call("cublasLtMatmulAlgoConfigSetAttribute", _C.byref(algo), attr, _C.byref(value), _C.sizeof(value))
        weight = st["qkv"]
        n, k = weight.shape
        assert weight.is_contiguous() and weight.data_ptr() % 128 == 0
        layouts = _graft_layouts(lt_tn, 256, n, k, "tn", 1, False)
        heur = _GraftHeur()
        code = lt_tn.lib.cublasLtMatmulAlgoCheck(lt_tn.handle, lt_tn.desc, *layouts, layouts[2], _C.byref(algo),
                                                 _C.byref(heur))
        status[("qkv", 256)] = (code, heur.workspaceSize)
        if not code and heur.workspaceSize <= lt_tn.workspace.numel():
            plans[("qkv", 256)] = _GraftPlan(lt=lt_tn, layouts=layouts, algo=algo, weight=weight, n=n, fp32=False)
    st["graft_lt"] = plans
    st["graft_lt_status"] = {f"{k[0]}:{k[1]}": v for k, v in status.items()}


def _graft_lt_run(x, plan):
    x = x.contiguous()
    assert x.data_ptr() % 16 == 0
    out = torch.empty((x.shape[0], plan["n"]), device=x.device,
                      dtype=torch.float32 if plan["fp32"] else torch.bfloat16)
    lt = plan["lt"]
    a, b, c = plan["layouts"]
    lt.call("cublasLtMatmul", lt.handle, lt.desc, _C.byref(lt.alpha), _GP(plan["weight"].data_ptr()), a,
            _GP(x.data_ptr()), b, _C.byref(lt.beta), _GP(out.data_ptr()), c, _GP(out.data_ptr()), c,
            _C.byref(plan["algo"]), _GP(lt.workspace.data_ptr()), lt.workspace.numel(),
            _GP(torch.cuda.current_stream().cuda_stream))
    return out


_vj_dense = dense


def dense(x, w, T, state=None, name=None):
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)


# ---- grafted boundary / route launches ----
def _graft_al(*ts):
    for t in ts:
        assert t.is_contiguous() and t.data_ptr() % 16 == 0, (t.shape, t.data_ptr() % 16)


def _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):
    """Returns True when a grafted kernel handled the boundary, else False (caller runs the base kernel)."""
    g = st["graft"]
    if S != 1:
        return False
    br = P.view(T, -1)
    width = br.shape[1]
    pfx = "2" if post else "1"
    if T <= 16 and br.dtype == torch.bfloat16:
        _graft_al(br, stream, s_out, h_out)
        if not post and width == HIDDEN:
            _graft_glue_module().b1(br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out,
                                    router_out if router_out is not None else s_out, router_out is not None)
            return True
        if post and width == 1024 and nu == 1024:
            _graft_b2_module().b2s(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        if post and width == HIDDEN and nu == HIDDEN:
            _graft_glue_module().b2c(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        return False
    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                      g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,
                                      num_warps=32, launch_pdl=True)
        return True
    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                  num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
    return False


def _graft_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if not _graft_boundary(P, S, T, data["stream"], state, False, s1, h, router_out=hr):
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                         WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        if T in (16, 32):
            nw, parts = {16: (8, 1), 32: (8, 1)}[T]
            _sl_tile_route[(T, parts + 1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=True, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=True)
        else:
            _graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        if T in (16, 32):
            nw, parts = {16: (8, 1), 32: (8, 1)}[T]
            _sl_tile_route[(T, parts + 1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=False, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=True)
        else:
            _graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if not _graft_boundary(P, S, T, s1, state, True, s2, out, nu=nu):
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids


_vj_after_attention = _after_attention


def _after_attention(data, state, o_in, gate, T):
    if "graft" in state and T in _GRAFT_GLUE_T and state.get("compose_graft_enabled", {}).get(T, T in (1, 8, 128, 256)):
        return _graft_after_attention(data, state, o_in, gate, T)
    return _vj_after_attention(data, state, o_in, gate, T)


_vj_kernel_fn = kernel_fn


def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None
    if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
        qkv = _rq_gemm(state, "qkv", data["attn_input"], T)
        if qkv is None:
            qkv = _graft_lt_run(data["attn_input"], plan)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    return _vj_kernel_fn(data, state)


_vj_prepare = prepare


def prepare(weights):
    st = _vj_prepare(weights)   # the base tuner runs first, on the base kernels (no "graft" keys yet)
    if _GRAFT_GLUE_T:
        # pin the o projection of the grafted small-T loads to bf16_proj's kernel (BF16 [1, T, 8192]), the base's
        # usual pick and the input the cluster boundary takes (the tuner flipped (1, "o") to cuBLAS once, gpu03)
        if st.get("bfp") is not None:
            for T in _GRAFT_GLUE_T:
                if T <= 16:
                    st["mm_pick"][(T, "o")] = "bfp"
        g = {}
        for i, name in ((1, "attn_residual_gate"), (2, "ffn_residual_gate")):
            gw = weights[name].float() / 0.3
            beta = torch.sigmoid(gw)
            g["alpha%d" % i] = torch.sqrt(torch.clamp(torch.sigmoid(-gw) * (1. + beta), min=1.e-3)).contiguous()
            g["beta%d" % i] = beta.contiguous()
        g["pre1"] = (weights["pre_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        g["post2"] = (weights["post_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        g["pre2"] = (weights["next_pre_attn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        for v in g.values():
            assert v.data_ptr() % 128 == 0, v.data_ptr() % 128
        st["graft"] = g
        _graft_glue_module()
        _graft_b2_module()
    if _GRAFT_LT_T:
        _graft_lt_plan(st)
    torch.cuda.synchronize()
    return st


def dl_pair_prewarm():
    _graft_glue_module()
    _graft_b2_module()

# MoE physical helper warp zero: only the 9..16-token MID kernel.
_MA_ALT_SRCS["moe_helper0"] = '\n// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.\n#include <cuda_bf16.h>\n#include <type_traits>\n#include <cuda_runtime.h>\n#include <stdint.h>\n#define OKC_DEBUG 0\n#define OKC_USE_TMA 1\n#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores\n#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0\'s barrier inits\n#define OKC_WGSFINSS 1  // WGS: the final pass forms each route\'s sum of squares (the phase-B epilogue only writes y)\n#ifndef OKC_FINCOL\n#define OKC_FINCOL 1\n#endif\n#define OKC_WGMIN 9\n#define OKC_XPRE 1\n#define FAST1_ARCH 1\n#ifndef OKC_FAST1\n#define OKC_FAST1 3\n#endif\n#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)\n#ifndef OKC_CPFF\n#define OKC_CPFF 1\n#endif\n#ifndef OKC_FCOSTBB\n#define OKC_FCOSTBB OKC_FCOSTB\n#endif\n#define OKC_FCOSTSB OKC_FCOST\n#define OKC_DYN 1\n#ifndef OKC_DYNK\n#define OKC_DYNK 4\n#endif\n#define OKC_TDYN 0\n#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)\n#define OKC_FT_T 64\n#ifndef OKC_SPLIT\n#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)\n#endif\n#ifndef OKC_TINY\n#define OKC_TINY 0\n#endif\n#ifndef OKC_NW\n#define OKC_NW 8\n#endif\n#ifndef OKC_EARLYN\n#define OKC_EARLYN 1\n#endif\n#define OKC_BIGM 64\n#define OKC_XPF 0\n#ifndef OKC_CPF\n#define OKC_CPF 1\n#endif\n#define OKC_FATSTATIC 1\n#define OKC_FCOST 40\n#ifndef OKC_FCOSTB\n#define OKC_FCOSTB 40\n#endif\n#ifndef OKC_WS\n#define OKC_WS 0  // tiny build: one extra helper warp issues the ring\'s TMA copies and runs every skinny epilogue\n#endif\n#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped\n#ifndef OKC_WGS\n#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp\n#endif\n#define SILU_DIV(a, b) __fdividef((a), (b))\n#ifndef OKC_UWARP\n#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers\n#endif\n#if OKC_UWARP\n#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)\n#else\n#define WARP_IDX(t) ((t) >> 5)\n#endif\n#ifndef OKC_PRMTSC\n#define OKC_PRMTSC 0\n#endif\n#ifndef OKC_FINPF\n#define OKC_FINPF 0   // skinny builds: final item\'s topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)\n#endif\n#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp\n#define OKC_NOLOAD 0\n\nnamespace okc {\n\n#ifndef OKC_L2H\n#define OKC_L2H 1\n#endif\n#if OKC_L2H\n// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in\n// the same call (evict last). Hints only: the math and the memory traffic\'s results are unchanged.\n#define L2H_SUFFIX ".L2::cache_hint"\n#define L2H_OPERAND ", %4"\n#define L2H_ARG , "l"(0x12F0000000000000ull)\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {\n  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");\n}\n__device__ __forceinline__ void st_keep_f32(float* a, float v) {\n  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");\n}\n#else\n#define L2H_SUFFIX ""\n#define L2H_OPERAND ""\n#define L2H_ARG\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }\n__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n#endif\n\nconstexpr int HD = 4096, ID = 4096;\nconstexpr int NW = OKC_NW, NT = NW * 32;\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\n#define TIDX ((int)((threadIdx.x + NT) % NTL))\nconstexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers\n#if OKC_WS || OKC_WGS\n// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)\n#define CTA_SYNC() asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory")\n#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL) : "memory")\n#else\n#define CTA_SYNC() __syncthreads()\n#endif\nconstexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)\n#define OKC_SKT 8\nconstexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)\nconstexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\nstatic_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\nconstexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)\nconstexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks\nconstexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;\nconstexpr int RB_A_LOG = 9, RB_B_LOG = 8;\nstatic_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");\nconstexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A\n// dynamic tail: the last pool units of a phase\'s skinny work are claimed at run time (OKC_DYNK per CTA)\n__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }\nstatic_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");\nstatic_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)\nconstexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;\nconstexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)\nconstexpr int KS = 256;         // fat: K per x stage (4 chunks)\nconstexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)\nconstexpr int NST = 3;          // fat x stages\nconstexpr float TWO62 = 4.611686018427388e18f;\n\nstruct Params {\n  const uint4* wA; const uint4* sA; const float* gsA;\n  const uint4* wB; const uint4* sB; const float* gsB;\n  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here\n  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)\n  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)\n  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)\n  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]\n  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]\n  int* xready;                           // WGS: per job, rows of xA written\n  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)\n  uint32_t zero;  // always 0 (opaque to the compiler)\n  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;\n  int M, lo, nloc;\n  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit\n  int* doneA;      // [JMAX]\n  __nv_bfloat16* inter;  // [EMAX, ID]\n  float* ybuf;     // [EMAX, HD]\n  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1\n  float* ss1;      // [EMAX]\n  __nv_bfloat16* out;\n  unsigned long long* dbg;\n};\n\nstruct Smem {\n  uint32_t lut[256];\n#if !OKC_WGS\n  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums\n#else\n  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)\n#endif\n#if OKC_WS\n  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free\n  int wsr[4];                          // WS: this CTA\'s phase A and phase B unit ranges\n#endif\n#if OKC_WGS\n  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty\n  int wglast[4];      // WGS: this warpgroup completed a split unit\n  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job\n  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job\n  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)\n  uint32_t wtot;      // WGS: phase-A total cost\n  uint32_t wtotB;     // WGS: phase-B total cost\n#else\n  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)\n#endif\n#if OKC_WGS\n  int wgr[4];         // WGS: this CTA\'s phase A and phase B unit ranges\n#endif\n  int seq, fseq, preissued, preissuedB;\n  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue\'s job -> expert map)\n  void* xbuf_ptr;\n  int cnt[64], off[64];\n  int wc[NW][32];\n  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];\n  int njF, njS, unit;\n  int gen;          // the grid barrier\'s generation at launch start (+1 per launch): picks this launch\'s ss copy\n  float* ssp;       // this launch\'s ss copy\n  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)\n  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)\n  int unit_next, unit_next2;\n  signed char eid[EMAX];\n  short ent[EMAX];   // sorted entries (token*8+slot) by expert\n  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n\n// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide\n// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up\n__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {\n  const int p = a * c;\n  int q = __float2int_rz((float)p * rG);\n  if (q * G > p) --q;\n  else if ((q + 1) * G <= p) ++q;\n  return q;\n}\n__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)\n__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n\n__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,\n                                         uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {\n  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);\n  int sz = valid ? 16 : 0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(s), "l"(gmem), "r"(sz));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n__device__ __forceinline__ unsigned long long gtimer() {\n  unsigned long long t;\n  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));\n  return t;\n}\n#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\nextern __shared__ __align__(16) unsigned char smem_raw[];\n__device__ __forceinline__ float* ss_cur();  // this launch\'s ss copy (defined after Smem users)\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));\n  return v;\n}\n\n// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).\n// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,\n// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.\n__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n  constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;\n  r1 = __funnelshift_r(q, q, 6) & M;\n  r2 = __funnelshift_r(q, q, 10) & M;\n  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:\n  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)\n  r1 = (q * 16u) & M;\n  r2 = __funnelshift_l(q, q, 10) & M;\n  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);\n  return;\n  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe\n  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates\n  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);\n}\n\n\n// ---------------------------------------------------------------- routing prologue (every CTA, identical result)\n__device__ void early_issue(const Params& p, Smem& S);\n__device__ void route(const Params& p, Smem& S) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int E = p.M * 8;\n  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;\n  const int i0 = warp * CH, i1 = min(E, i0 + CH);\n  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;\n#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)\n  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores\n    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;\n    int4 ev[LPT];\n    const int E4 = E >> 2;\n#pragma unroll\n    for (int k = 0; k < LPT; ++k) {\n      const int i4 = tid + k * NT;\n      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);\n    }\n#pragma unroll\n    for (int k = 0; k < LPT; ++k) {\n      const int i4 = tid + k * NT;\n      if (i4 < E4) {\n        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};\n        char4 c;\n        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);\n        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);\n        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);\n        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);\n        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;\n      }\n    }\n  }\n#else\n#pragma unroll 8\n  for (int i = tid; i < E; i += NT) {\n    int e = __ldg(p.topk_ids + i) - p.lo;\n    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);\n  }\n#endif\n  CTA_SYNC();\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n  }\n  CTA_SYNC();\n  if (warp == 0) {\n    int c[NW];\n    int tot = 0;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }\n    int incl = tot;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }\n    int off = incl - tot;\n    S.cnt[lane] = tot;\n    S.off[lane] = off;\n    int o2 = off;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }\n    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each\n#if OKC_SPLIT\n    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");\n    const int nfj = 0;\n    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights\n#else\n    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;\n    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;\n#endif\n    int fi = nfj, si = nsj;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);\n      if (lane >= d) { fi += v1; si += v2; }\n    }\n    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);\n    int fj = fi - nfj, sj = nfTot + si - nsj;\n    for (int q = 0; q < nfj; ++q) {\n      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);\n    }\n#if OKC_SPLIT\n    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }\n#else\n    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }\n#endif\n    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }\n    __syncwarp();\n#if !OKC_WGS  // (the WGS build has no fat queue)\n    // fat queue order: largest n-tile count first (LPT), stable by job index\n    int pos = 0;\n    for (int key = FT_T / 8; key >= 1; --key) {\n      for (int f0 = 0; f0 < nfTot; f0 += 32) {\n        const int f = f0 + lane;\n        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;\n        const unsigned bal = __ballot_sync(0xffffffffu, m);\n        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;\n        pos += __popc(bal);\n      }\n    }\n#endif\n  }\n  CTA_SYNC();\n  early_issue(p, S);\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0) {\n      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));\n      S.ent[q] = (short)i;\n      S.pos[i] = (short)q;\n    } else if (i < i1) {\n      S.pos[i] = -1;\n    }\n    __syncwarp();\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n    __syncwarp();\n  }\n  CTA_SYNC();\n}\n\n\n\n// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)\n// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile\n__device__ __forceinline__ float probe_q(float v) {\n  return v;\n}\n__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    if (tk < ntok) {\n      float gt = r[h] * gsc, up = r[2 + h] * gsc;\n      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));\n    }\n  }\n}\n// returns per-token partial sum of squares (2 values) for this lane\'s two tokens\n__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,\n                                      float* sq) {\n  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;\n    sq[h] = v0 * v0 + v1 * v1;\n    if (tk < ntok) {\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);\n    }\n  }\n}\n__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);\n  }\n  if (g == 0) {\n#pragma unroll\n    for (int h = 0; h < 2; ++h)\n      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);\n  }\n}\n\n// ---------------------------------------------------------------- the persistent kernel\n\n\n// ---------------------------------------------------------------- skinny path: shared-memory weight ring\n// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.\n// Units of the CTA\'s range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).\n// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.\nconstexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;\n\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n#if OKC_PRMTSC\n// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the\n// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes\n// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.\n#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)\n// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the\n// ALU pipe, which the FP4 decode saturates\n#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)\n#endif\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n .reg .pred p;\\n WAIT_%=:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra WAIT_%=;\\n}\\n"\n      ::"r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n  const size_t blk = (size_t)S.jE[j] * rbs + rb;\n  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,\n                                            uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], st = S.jS[j];\n  {\n    const int r = g;\n    if (r < cnt) {\n      const __nv_bfloat16* row;\n      if (PB) row = p.inter + (size_t)(st + r) * ID;\n      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) {\n        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);\n        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);\n      }\n    } else {\n#pragma unroll\n      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n    }\n  }\n}\n\nconstexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16\ntemplate <bool PB>\n__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,\n                                             __nv_bfloat16* x1s) {\n  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4\n  const int cnt = S.jC[j], st = S.jS[j];\n  for (int i = lane; i < 8 * 64; i += 32) {\n    const int r = i >> 6, c = i & 63;\n    uint4 v = make_uint4(0, 0, 0, 0);\n    if (8 + r < cnt) {\n      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n      v = PB ? __ldcg(src) : __ldg(src);\n    }\n    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;\n  }\n  __syncwarp();\n}\n\n#if SKDYN\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows\n  const int rsh = PB ? RB_B_LOG : RB_A_LOG;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  // cross-phase prefill: with no fat jobs, phase A\'s last ring refills already fetch this CTA\'s first B units\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight\n  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;\n  bool exh = poolBase >= nSall;\n  int* pctr = p.ctr + (PB ? 1 : 0);\n  auto take = [&]() -> int {\n    if (snext < v1) return snext++;\n    if (exh) return -1;\n    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);\n    const int v = poolBase + cl;\n    if (v >= nSall) { exh = true; pend = -2; return -1; }\n    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill\n    if (!PB) {\n      const int jj = njF + (v >> rsh);\n      S.dync[jj] += 1;\n      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);\n    }\n    return v;\n  };\n  auto refill = [&](int sl) {\n    const int v = take();\n    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }\n    else if (cpf && preB < nB) {\n      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);\n      S.uslot[sl] = BFLAG | (v0B + preB);\n      ++preB;\n    } else {\n      S.uslot[sl] = -1;\n    }\n  };\n  if (tid == 0)\n    for (int i = pre; i < OKC_NS; ++i) {\n      const int sl = (seq0 + i) % OKC_NS;\n      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }\n      else refill(sl);\n    }\n  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) {\n    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);\n    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);\n    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n  }\n  int nproc = 0;\n  for (int i = 0;; ++i) {\n    int v;\n    if (i < nU) {\n      v = v0 + i;  // static units: no slot lookup\n    } else {\n      v = S.uslot[(seq0 + i) % OKC_NS];\n      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B\'s prefetched units follow)\n      v &= ~BFLAG;\n    }\n    const int j = njF + (v >> rsh), rb = v & (rbs - 1);\n    float nwn0 = 0.f, nwn1 = 0.f;\n    int vn = -1;\n    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];\n    if (PB && tid < 128 && vn >= 0) {\n      const int rbq = (vn & ~BFLAG) & (rbs - 1);\n      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n#if OKC_NW == 16\n    // a warp covers half of an 8-warp K slice: its 4 chunks\' scale bytes are 8 of the slice lane\'s 16\n    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);\n    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);\n#else\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n#endif\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();  // slot sl fully consumed; partials visible\n    if (tid == 0) {\n      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }\n      else refill(sl);\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float gt = s0 * gsc, up = s1 * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n    ++nproc;\n  }\n  seq = seq0 + nproc;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      if (nU > 0) {\n        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);\n        for (int jj = jf; jj <= jl; ++jj) {\n          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n          atomicAdd(p.doneA + jj, b - a);\n        }\n      }\n      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units\n        const int k = S.dync[jj];\n        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#else\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  if (nU <= 0) return;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  // cross-phase prefill: with no fat jobs, phase A\'s last ring refills already fetch this CTA\'s first B units\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  for (int i = pre; i < OKC_NS; ++i) {\n    const int sl = (seq0 + i) % OKC_NS;\n    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }\n  }\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    float nwn0 = 0.f, nwn1 = 0.f;\n    if (PB && tid < 128 && i + 1 < nU) {\n      const int rbq = (v + 1) % rbs;\n      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) {\n#pragma unroll\n      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();  // slot sl fully consumed; partials visible\n    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i + OKC_NS - nU < nB) {\n      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);\n      ++preB;\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float gt = s0 * gsc, up = s1 * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n  }\n  seq = seq0 + nU;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;\n      for (int jj = jf; jj <= jl; ++jj) {\n        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n        atomicAdd(p.doneA + jj, b - a);\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#endif\n\n// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens\n// ---------------------------------------------------------------- fat unit through the smem ring\n// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x\n// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile\n// for the stage is staged by cp.async into two buffers that live in slots 3..4.\nconstexpr int NSF = OKC_NSF;\ntemplate <bool PB>\n__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {\n  const int nub = PB ? UB_B : UB_A;\n  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;\n  if (TIDX == 0) {\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n  }\n}\n\n// ---------------------------------------------------------------- fat units: one continuous stream per CTA\n// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps\n// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit\'s weights).\n// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem\n__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);\n}\n__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }\ntemplate <int N>\n__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory"); }\ntemplate <int NTT> struct WG;\ntemplate <> struct WG<1> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<2> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<3> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<4> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<5> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<6> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<7> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<8> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\n\n// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from\n// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {\n  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads\n  // k block cc at byte offset 32 st\n  const int g = lane >> 2;\n  const uint32_t xb = smem_u32(xs);\n  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired\n  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const int sb = cc & 1;\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);\n    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact\n      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2\n      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)\n      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);\n      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);\n#define FMG0(r) bmul2((r), sg0)\n#define FMG80(r) bmul2((r), sg80)\n#define FMG1(r) bmul2((r), sg1)\n#define FMG81(r) bmul2((r), sg81)\n      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;\n      dq4((&w0.x)[st], r0, r1, r2, r3);\n      dq4((&w1.x)[st], q0, q1, q2, q3);\n      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);\n      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);\n      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);\n      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);\n#undef FMG0\n#undef FMG80\n#undef FMG1\n#undef FMG81\n    }\n    wg_fence();\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);\n      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);\n      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);\n    }\n    wg_commit();\n    if (cc >= 1) {\n      wg_wait<1>();  // the previous chunk\'s group retired: its register set may be rewritten\n#pragma unroll\n      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];\n    }\n  }\n  wg_wait<0>();\n#pragma unroll\n  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];\n  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f\n}\n\n// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {\n  const int g = lane >> 2, t = lane & 3;\n  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);\n#if OKC_NW == 8\n  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);\n#endif\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n#if OKC_NW == 8\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n#endif\n#if OKC_PRMTSC\n    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);\n#if OKC_NW == 8\n    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);\n#endif\n#else\n    const int sh = (cc & 1) * 16;\n    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;\n    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;\n    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];\n    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];\n#endif\n    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;\n#pragma unroll\n    for (int sp = 0; sp < 2; ++sp) {\n      uint32_t a0[2][4], a1[2][4];\n      (void)a1;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int s2 = 2 * sp + h;\n        uint32_t r0, r1, r2, r3;\n        dq4((&w0.x)[s2], r0, r1, r2, r3);\n        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);\n#if OKC_NW == 8\n        dq4((&w1.x)[s2], r0, r1, r2, r3);\n        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);\n#endif\n      }\n#pragma unroll\n      for (int nt = 0; nt < NTT; ++nt) {\n        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);\n        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);\n#endif\n        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);\n#endif\n      }\n    }\n  }\n}\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int g = lane >> 2, t = lane & 3;\n  int* ctr = p.ctr + (PB ? 1 : 0);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n  uint64_t* fb = S.bars + 5;\n  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);\n  constexpr int NKS = HD / KS;  // 16\n  int u = blockIdx.x;\n  (void)ctr;\n  if (u >= nF) return;\n  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)\n  (void)ubs;\n  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };\n  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete\n    if (PB) {\n      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }\n      CTA_SYNC();\n    }\n  };\n  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };\n  auto stage_x = [&](int j, int ks, int xsl) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int ntl = (cnt + 7) >> 3;\n    const int nrow = ntl * 8;\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n    const int per = nrow * (KS / 8);\n    if (is_nat(j)) {\n      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)\n      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);\n      for (int i = tid; i < per; i += NT) {\n        const int r = i >> 5, ch = i & 31;\n        const bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;\n        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);\n      }\n    } else {\n      for (int i = tid; i < per; i += NT) {\n        int r = i / (KS / 8), c = i - r * (KS / 8);\n        bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;\n        cp_async16(xs + r * XP_FT + c * 8, src, v);\n      }\n    }\n    cp_commit();\n  };\n  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)\n  const __nv_bfloat16* xsrc[8];\n  uint32_t xdst[8];\n  uint32_t xval = 0;\n  int xn = 0;\n  auto prep_x = [&](int j) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);\n    xval = 0;\n    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;\n#pragma unroll\n    for (int k = 0; k < 8; ++k) {\n      const int i = tid + k * NT;\n      const int r = i >> 5, c = i & 31;\n      const bool v = i < per && r < cnt;\n      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;\n      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;\n      xdst[k] = r * XP_FT + c * 8;\n      if (v) xval |= 1u << k;\n    }\n  };\n  auto stage_xp = [&](int ks, int xsl) {\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n#pragma unroll\n    for (int k = 0; k < 8; ++k)\n      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);\n    cp_commit();\n  };\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  int f = S.fseq;  // ring position of the current unit\'s k-stage 0\n  int xq = 0;      // x stage parity counter\n  int j, ub;\n  job_of(u, j, ub);\n#pragma unroll\n  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));\n  wait_A(j);\n  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);\n  while (true) {\n    const int un = u + gridDim.x;\n    int jn = 0, ubn = 0;\n    if (un < nF) job_of(un, jn, ubn);\n    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];\n    const int ntile = (cnt + 7) >> 3;\n    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)\n    const bool nat = is_nat(j);\n    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;\n    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];\n#pragma unroll\n    for (int i = 0; i < FT_T / 8; ++i)\n#pragma unroll\n      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }\n    for (int ks = 0; ks < NKS; ++ks) {\n      if (OKC_XPRE) {\n        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }\n        else cp_commit();\n      } else {\n        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }\n        else cp_commit();\n      }\n      cp_wait<1>();\n      const int q = f + ks, sl = q % NSF;\n      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));\n      if (nat) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");  // cp.async x writes -> wgmma reads\n      CTA_SYNC();  // x stage visible to every warp\n      const unsigned char* slot = ring + sl * UNIT_BYTES;\n      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);\n      if (nat) {\n        switch (ntc) {\n          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n        }\n      } else {\n        switch (ntc) {\n          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;\n          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;\n        }\n      }\n      CTA_SYNC();  // slot sl and the x stage consumed\n      ++xq;\n      // refill: the stage NSF ahead, in this unit or the next one\n      const int kn = ks + NSF;\n      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);\n      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));\n    }\n    // epilogue of unit u (registers -> global)\n    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;\n    if (!PB) {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);\n#if OKC_NW == 8\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);\n#endif\n        }\n      __threadfence();\n      CTA_SYNC();\n      if (tid == 0) atomicAdd(p.doneA + j, 1);\n    } else {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          float sq0[2], sq1[2];\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);\n#if OKC_NW == 8\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);\n          sq0[0] += sq1[0]; sq0[1] += sq1[1];\n#else\n          (void)sq1;\n#endif\n          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);\n        }\n    }\n    f += NKS;\n    if (un >= nF) break;\n    u = un; j = jn; ub = ubn;\n  }\n  cp_wait<0>();\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  if (tid == 0) S.fseq = f;\n  CTA_SYNC();\n}\n\ntemplate <bool PB>\n__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  fat_stream<PB, false>(p, S, xbuf, nF, ubs);\n}\n\n\n\n// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)\n__device__ void early_issue(const Params& p, Smem& S) {\n#if FAST1_ON || OKC_WGS\n  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring\n#endif\n  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region\n    if (TIDX == 0) S.preissued = 0;\n    return;\n  }\n  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);\n  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in\n  // skinny_ring issues the rest after routing, so every CTA\'s first unit is served ahead of the second ones\n  const int n = min(OKC_EARLYN, v1 - v0);\n  if (TIDX == NT - 32) {\n    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }\n    S.preissued = n > 0 ? n : 0;\n  }\n}\n\n// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost\n#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper\n// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit\'s\n// epilogue, so no CTA-wide barrier sits in the per-unit loop.\n// the last WSBLK units keep their partial buffers until the compute warps\' own epilogues: WSBLK > WSNR deadlocks the\n// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)\nstatic_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");\n#define OKC_WSRF 0\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\ntemplate <bool PB>\n__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  const int seq0 = seq;\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  float lgs = 0.f;                             // the last phase-B unit\'s epilogue constants (warp 0)\n  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs;\n    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {\n      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);\n      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);\n      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);\n    }\n    if (j != curj) {\n      curj = j;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n    }\n    const int q = seq0 + i;\n    const int sl = q % OKC_NS;\n    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {\n        uint32_t r0, r1, r2, r3;\n        dq4(ww[st], r0, r1, r2, r3);\n        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB: the compute warps keep the base loop\'s CTA barrier per unit (slot consumed + partials visible), thread 0\n    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop\n    const int bb = q % NRB;\n    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer\'s previous unit\n    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);\n      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n    }\n  }\n  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA\'s last units: epilogues here, not behind the helper\'s queue\n    const int ul = nU - 1 - warp;\n    const int q = seq0 + ul, bb = q % NRB;\n    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n    const int gg = lane >> 2, tt = lane & 3;\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    const int n0 = rb * 16 + gg;\n    const float gsc = lgs * TWO62;\n    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = 2 * tt + h;\n      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n      float sq = v0f * v0f + v1f * v1f;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n      }\n      sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n    }\n  }\n  seq = seq0 + nU;\n}\n\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {\n  const int sl = q % OKC_NS;\n  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);\n  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);\n}\n\n__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n  const int lane = TIDX & 31;\n  WS_JOIN();  // routing tables and this CTA\'s ranges are ready\n  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  const int njF = S.njF;\n  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n  const int gg = lane >> 2, tt = lane & 3;\n  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;\n  for (int q = 0; q < Nh; ++q) {\n    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q\'s slot once every compute warp released it\n      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);\n    }\n    const bool PB = q >= nA;\n    const int v = PB ? b0 + (q - nA) : a0 + q;\n    const int rbs = PB ? RB_B : RB_A;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    const int bb = q % NRB;\n    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    if (!PB) {\n      const float gsc = p.gsA[ce] * TWO62;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        if (tk < cnt) {\n          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      }\n      if (q == nA - 1) {  // this CTA\'s phase A is done: credit its units to the jobs\' counters\n        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n          for (int jj = jf; jj <= jl; ++jj) {\n            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);\n            atomicAdd(p.doneA + jj, b - a);\n          }\n        }\n      }\n    } else {\n      const float gsc = p.gsB[ce] * TWO62;\n      const int n0 = rb * 16 + gg;\n      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n  }\n  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n\ntemplate <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {\n  const float rG = __fdividef(1.f, (float)G);\n  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;\n  auto fcost = [](long long k) -> long long {\n    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B\'s skinny units are slower, so its ratio is lower)\n    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);\n    return k > 0 ? (long long)fc * k : 0;\n  };\n  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level\n  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;\n  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;\n  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };\n  const long long WG = W(G);\n  if (nF > 0 && WG > 0) {\n    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))\n    const float sc = __fdividef((float)nS, (float)WG);\n    auto split = [&](int x) -> int {\n      if (x >= G) return nS;\n      const int v = __float2int_rz((float)W(x) * sc);\n      return v < nS ? v : nS;\n    };\n    v0 = split(c); v1 = split(c + 1);\n  } else {\n    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);\n  }\n}\n\n#if OKC_WGS\n// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)\n// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,\n// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales\n// [grp][warp][g][h][j]; the stage\'s x slice (the job\'s token rows, rounded up to 8, x 256 K as wgmma B core matrices\n// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16\n// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).\n// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a\n// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.\n#define OKC_WGSNX 3                             // x ring slots\nconstexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)\nconstexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage\'s last wgmma retired)\nconstexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job\nconstexpr int WGS_PIECE = 9216;                 // one warpgroup\'s weights in a stage\nconstexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K\nconstexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices\nstatic_assert(BX_E + WGS_NX <= 24, "WGS barriers");\nconstexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)\nconstexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]\nconstexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs\n__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {\n  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;\n}\n__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {\n  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;\n}\n// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at\n// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each\n#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)\n#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\nconstexpr unsigned long long wgs_pack8(const int (&c)[8]) {\n  unsigned long long r = 0;\n  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);\n  return r;\n}\nconstexpr int WGS_CA[8] = {OKC_WGSA};\nconstexpr int WGS_CB[8] = {OKC_WGSB};\nconstexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);\nstatic_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");\n__device__ __forceinline__ int wgs_cost(int pb, int cnt) {\n  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);\n}\n#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)\n#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time\n#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather\n#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n#define OKC_WGSFCI 3\n#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)\n__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {\n  (void)tag;\n  mbar_wait(bar, parity);\n}\ntemplate <int NTT> struct WGSMMA;\ntemplate <> struct WGSMMA<1> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<2> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<3> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<4> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<5> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<6> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<7> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<8> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\n// two producer warps: the weights warp streams every stage\'s 36 KB through the weight ring (TMA, lane 0), the x warp\n// gathers every stage\'s x slice into the x ring (cp.async) and, in phase B, first waits for the job\'s phase-A units;\n// so the weights run ahead by the whole ring even while x waits\n// Phase A runs a CTA\'s trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit\'s\n// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase\n// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the\n// order (its first stages wait for phase A\'s units in order).\n__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {\n  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range\'s last stage\n  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;\n}\n__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)\n__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\n  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  if ((TIDX & 31) != 0) return;\n  int ws = 0, n = 0;\n  uint32_t wph = 0;\n#pragma unroll 1\n  for (int pb = 0; pb < 2; ++pb) {\n    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);\n    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job\n    const unsigned char* base = pb ? p.wBs : p.wAs;\n#pragma unroll 1\n    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {\n      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)\n      const int u = gs >> 4, st = gs & 15;\n      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);\n      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n      uint64_t* bar = S.bars + BW_F + ws;\n      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n  }\n}\n// x producer: one bulk copy per stage (lane 0): the job\'s x slice for the stage is contiguous in xA (phase A, written\n// by every CTA\'s gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B\nconstexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound\n__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {\n  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the\n  // stage ranges while the consumers gather phase A\'s x\n  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");\n  if ((TIDX & 31) == 0) wgs_ranges(p, S);\n  __syncwarp();\n  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  if ((TIDX & 31) != 0) return;\n  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;\n  int xs = 0, n = 0;\n  uint32_t xph = 0;\n#pragma unroll 1\n  for (int pb = 0; pb < 2; ++pb) {\n    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);\n    const int ubl = pb ? 4 : 5;\n    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;\n    int curj = -1, npad = 8;\n    const __nv_bfloat16* jb = xsrc;\n#pragma unroll 1\n    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {\n      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)\n      const int u = gs >> 4, st = gs & 15;\n      const int j = u >> ubl;\n      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);\n      if (j != curj) {\n        curj = j;\n        npad = ((S.jC[j] + 7) >> 3) << 3;\n        jb = xsrc + (size_t)S.jrow[j] * HD;\n        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered\n      }\n      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them\n      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }\n      uint64_t* bar = S.bars + BX_F + xs;\n      const uint32_t bytes = (uint32_t)npad * 512u;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"\n                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }\n    }\n  }\n}\n// phase A\'s x gather (all consumer threads of every CTA, once, at the start): this CTA\'s share of the jobs\' padded rows\n// copied into xA\'s per-stage wgmma layout, then counted into xready (padding rows are counted, not written)\n__device__ void wgs_gather(const Params& p, Smem& S) {\n  const int tid = TIDX, nj = S.njF + S.njS;\n  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);\n  int j = 0;\n  // all of a batch\'s row loads first, then the stores: one x row at a time was one dependent HBM round trip per row\n  // (the next row\'s load could not move above the previous row\'s store: they may alias), 1.3-7 us at bs256\n  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");\n  constexpr int GB = OKC_GATHB;\n  for (int qb = q0; qb < q1; qb += GB) {\n    uint4 v[GB];\n    uint4* d[GB];\n#pragma unroll\n    for (int k = 0; k < GB; ++k) {\n      const int q = qb + k;\n      d[k] = nullptr;\n      if (q < q1) {\n        while (S.jrow[j + 1] <= q) ++j;\n        const int r = q - S.jrow[j], cnt = S.jC[j];\n        if (r < cnt) {\n          const int ntt = (cnt + 7) >> 3;\n          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));\n          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);\n        }\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < GB; ++k)\n      if (d[k]) *d[k] = v[k];\n  }\n  CTA_SYNC();\n  if (tid == 0) {  // count this CTA\'s rows per job (release: the CTA\'s stores are ordered before by the barrier)\n    int jj = 0;\n    for (int q = q0; q < q1;) {\n      while (S.jrow[jj + 1] <= q) ++jj;\n      const int e = min(q1, S.jrow[jj + 1]);\n      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");\n      q = e;\n    }\n  }\n}\n// Work left from the previous segment, done after the next segment\'s first wgmma group (so the stores and atomics it\n// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split\n// unit\'s share and, if that completes the unit, read the summed partials back and run its epilogue.\nstruct WgsPend { int kind, u, slot, nst; };\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {\n  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];\n  if (!PB) {\n    // SwiGLU straight into xB (phase B\'s per-stage wgmma B layout) instead of row-major inter\n    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;\n    const float gsc = p.gsA[e] * TWO62;\n    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = nt * 8 + 2 * t + h;\n      if (tk < cnt) {\n        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;\n        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),\n                     __float2bfloat16(probe_q(v)));\n      }\n    }\n  } else {\n    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)\n    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;\n    const float gsc = p.gsB[e] * TWO62;\n    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = tok0 + h;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n      }\n    }\n  }\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {\n  if (pd.kind == 0) return;\n  const int tid = TIDX, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;\n  constexpr int UBL = PB ? 4 : 5;\n  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  if (pd.kind == 1) {\n    // release (cumulative over the warpgroup\'s epilogue stores ordered by the barrier); pairs with the producer\'s\n    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");\n    pd.kind = 0;\n    return;\n  }\n  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;\n  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase\n  if constexpr (DF) {\n  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the\n  // unit\'s last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA\'s end). The others just\n  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts\n  if (pd.kind == 2) {\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");\n    pd.kind = 0;\n    return;\n  }\n  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  pd.kind = 0;\n  } else {\n  if (wt == 0) {\n    int old;\n    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");\n    S.wglast[wg] = old + pd.nst == 16;\n  }\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  const bool last = S.wglast[wg];\n  pd.kind = 0;\n  if (!last) return;\n  }\n  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;\n  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;\n  // two n-tiles\' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load\n  // waits behind the previous tile\'s stores (they may alias), ntt serialized L2 round trips under full HBM load\n  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {\n    float4 v[2];\n#pragma unroll\n    for (int k = 0; k < 2; ++k)\n      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));\n#pragma unroll\n    for (int k = 0; k < 2; ++k)\n      if (nt0 + k < ntt) {\n        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);\n        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};\n        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);\n      }\n  }\n  if (wt == 0) *fc = 0;\n  if (!PB) {\n    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");\n  }\n}\n// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)\ntemplate <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,\n                                        int& seq, WgsPend& pd) {\n  const int tid = TIDX, lane = tid & 31;\n  const int warp = WARP_IDX(tid);\n  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;\n  constexpr int UBL = PB ? 4 : 5;\n  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)\n  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);\n  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)\n  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas\n  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time\n  constexpr int GS = OKC_WGSGS;\n  uint32_t af[2][GS][4];\n  int prev = -1;  // x slot the last committed group may still read\n  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot\'s offset (the address field cannot\n  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)\n  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);\n  const unsigned char* pc0 = ring + wg * WGS_PIECE;\n  auto stage = [&](auto first) {\n    constexpr bool FIRST = decltype(first)::value;\n    const int ws = seq % WGS_NS, xs = seq % WGS_NX;\n    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));\n    // codes and scales one 4-step group at a time (fewer live registers)\n    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n      const uint4 cwg = cwn;\n      const uint2 scg = scn;\n      if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);\n      } else if (lane == 0) {\n        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free\n      }\n#pragma unroll\n      for (int h = 0; h < 4 / GS; ++h) {\n        const int ci = lg * (4 / GS) + h;  // commit group index in the stage\n#pragma unroll\n        for (int jj = 0; jj < GS; ++jj) {\n          const int js = h * GS + jj;      // k16 step within the 4-step load group\n          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;\n          uint32_t r0, r1, r2, r3;\n          dq4(q, r0, r1, r2, r3);\n          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows\' scales in\n          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer\n          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;\n          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);\n        }\n        wg_fence();\n#pragma unroll\n        for (int jj = 0; jj < GS; ++jj) {\n          if (FIRST && ci == 0 && jj == 0)\n            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);\n          else\n            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],\n                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n        }\n        wg_commit();\n        wg_wait<1>();\n        if (ci == 0) {\n          if (prev >= 0) {  // the previous stage\'s last group just retired: its x slot is free\n            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n            prev = -1;\n          }\n        }\n        // the previous segment\'s deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG\'s\n        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)\n        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n      }\n    }\n    prev = xs;\n    ++seq;\n  };\n  stage(std::true_type{});\n#pragma unroll 1\n  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});\n  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }\n  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)\n    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;\n#pragma unroll\n    for (int nt = 0; nt < NTT; ++nt)\n      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),\n                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");\n    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,\n    // phase B the one holding stage 0\n    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};\n    return;\n  }\n  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n#pragma unroll\n  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {\n  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)\n  constexpr int UBL = PB ? 4 : 5;\n  WgsPend pd{0, 0, 0, 0};\n#pragma unroll 1\n  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {\n    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)\n      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);\n      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;\n    }\n    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));\n    // a unit split between CTAs: its fixup slot is the CTA holding its first stage\n    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));\n    const int ntt = (S.jC[u >> UBL] + 7) >> 3;\n    switch (ntt) {\n      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n    }\n    gs += s1 - s0;\n  }\n  wgs_flush<PB>(p, S, pd);\n}\n// cost-weighted stage ranges: per-stage cost by the job\'s n-tiles, a prefix per job and phase (S.wcp, S.wcpB)\n// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run\n// from a cold instruction cache at the start of every launch\n__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {\n  uint32_t acc = 0, accB = 0;\n  for (int j = 0; j < nj; ++j) {\n    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;\n    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;\n    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;\n  }\n  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;\n  S.wtot = acc; S.wtotB = accB;\n}\n// stage at cost position x (phase pb), the stage whose cost interval holds x\n__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {  if (nj == 0) return 0;  // achitaan: no local route (all experts remote) -> empty stage ranges; S.jC[0] is unset\n  const int* pre = pb ? S.wcpB : S.wcp;\n  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x\n  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }\n  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);\n  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);\n}\n__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);\n  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);\n}\n// this CTA\'s stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same\n__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {\n  const int nj = S.njF + S.njS;\n  wgs_prefix(S, nj);\n  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);\n  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}\n// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)\n__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  const int j = s >> (9 - pb);\n  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);\n  // largest c with floor(tot * c / G) < xe\n  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);\n  if (c > G - 1) c = G - 1;\n  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;\n  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;\n  return c;\n}\n#endif\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n  const int njF = S.njF, njS = S.njS;\n  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;\n  const int nS = njS * rbs, nF = njF * ubs;\n  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost\n#if OKC_TINY\n  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units\n#else\n  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }\n#endif\n  const int G = gridDim.x, c = blockIdx.x;\n  int v0, v1;\n  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool\n  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);\n  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A\'s last ring\n  // refills already fetch those units (same split function as phase B uses)\n  int cB0 = 0, cBn = 0;\n  bool cpf = false;\n  if (!PB) {\n    const int nFB = njF * UB_B;\n    if (c >= nFB) {\n      int b1;\n      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);\n      cBn = b1 - cB0;\n      cpf = OKC_CPFF || njF == 0;\n    }\n  }\n#if OKC_WS\n  (void)cpf; (void)cBn;\n  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n#else\n  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,\n                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);\n#endif\n}\n\n// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register\n// cap: spills)\n__device__ __forceinline__ float* ss_cur() {\n#if OKC_WGS\n  unsigned long long v;\n  asm volatile("ld.volatile.shared.u64 %0, [%1];\\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));\n  return reinterpret_cast<float*>(v);\n#else\n  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist\n#endif\n}\ntemplate <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {\n  Smem& S = *reinterpret_cast<Smem*>(smem_raw);\n  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);\n  const int tid = TIDX;\n  DBG(0);\n  // programmatic dependent launch: everything below reads the previous kernel\'s outputs (routing, x) or writes\n  // scratch, so wait for it here; without the launch attribute this returns at once\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#if OKC_WS\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n#if OKC_WGS\n  if (tid >= NT) {\n    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    return;\n  }\n#endif\n#if FAST1_ON\n  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA\'s first units follow\n  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path\n  if (tid >= NT - 32) {\n    const int ln = tid & 31;\n#if OKC_SPLIT\n    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)\n    S.fcnt[ln] = 0;\n    __syncwarp();\n    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {\n      const int i = i0 + ln;\n      int e = -1;\n      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n      const unsigned peers = __match_any_sync(0xffffffffu, e);\n      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n    }\n    __syncwarp();\n    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }\n    const int njob = __shfl_sync(0xffffffffu, jincl, 31);\n    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);\n    const float rG1 = __fdividef(1.f, (float)gridDim.x);\n    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n    const int n1 = min(OKC_FAST1, v11 - v01);\n    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];\n#pragma unroll\n    for (int i = 0; i < OKC_FAST1; ++i) {\n      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)\n      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;\n    }\n#elif OKC_F1EARLY\n    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the\n    // barriers while they are in flight (the inits used to wait behind the cold loads)\n    uint32_t m = 0;\n    const int E8 = p.M * 8;\n    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;\n    if (ln == 0) {\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    }\n    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)\n      const int e = __ldg(p.topk_ids + i) - p.lo;\n      if (e >= 0 && e < p.nloc) m |= 1u << e;\n    }\n    m = __reduce_or_sync(0xffffffffu, m);\n#endif\n    if (ln == 0) {\n#if OKC_SPLIT || !OKC_F1EARLY\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n#if OKC_SPLIT\n      const int v0 = v01, n = n1;\n#else\n      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)\n      const float rG = __fdividef(1.f, (float)gridDim.x);\n      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n      const int n = min(OKC_FAST1, v1 - v0);\n#endif\n      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n#if OKC_SPLIT\n#pragma unroll\n      for (int i = 0; i < OKC_FAST1; ++i) {\n        if (i >= n) break;\n        const int v = v0 + i;\n        const int e = fex[i];\n#else\n      for (int i = 0; i < n; ++i) {\n        const int v = v0 + i;\n        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);\n#endif\n        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;\n        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;\n        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");\n        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");\n      }\n      S.preissued = n > 0 ? n : 0;\n    }\n  }\n#endif\n  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)\n    asm volatile("prefetch.global.L2 [%0];\\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));\n  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)\n    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;\n    uint32_t b;\n    if (ex == 15 && mt == 7) {\n      b = 0xFFFFu;\n    } else {\n      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);\n      v *= 0x1p64f;\n      b = __float_as_uint(sgn ? -v : v) >> 16;\n    }\n    S.lut[i] = b | (b << 16);\n  }\n  if (tid == 0) {\n#if !FAST1_ON\n#if OKC_WGS\n    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),\n    // partial buffer free (the epilogue warpgroup)\n    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n#else\n    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n#endif\n    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    S.preissued = 0;\n#endif\n    S.seq = 0;\n    S.fseq = 0;\n    S.preissuedB = 0;\n    S.xbuf_ptr = xbuf;\n    // the generation can only change at this launch\'s grid barrier, which every CTA reaches after this point\n    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)\n    S.gen = gen;\n    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n  }\n#if SKDYN\n  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;\n#endif\n  route(p, S);\n#if OKC_WGS && OKC_WGSPR\n  __threadfence_block();\n  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges\n#endif\n  DBG(1);\n  {  // the previous launch\'s ss copy and barrier counter are the next launch\'s: zero them now (this CTA\'s share)\n    float* other = (S.gen & 1) ? p.ss : p.ss1;\n    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;\n    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n  }\n#if OKC_WS\n  if (tid == 0) {  // this CTA\'s unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)\n    const int G = gridDim.x, c = blockIdx.x;\n    const float rG = __fdividef(1.f, (float)G);\n    const int nA = S.njS * RB_A, nB = S.njS * RB_B;\n    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n#endif\n#if OKC_WGS\n  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)\n    const int nj = S.njF + S.njS;\n    int carry = 0;\n    for (int j0 = 0; j0 < nj; j0 += 32) {\n      const int j = j0 + tid;\n      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;\n      int incl = v;\n#pragma unroll\n      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }\n      if (j < nj) S.jrow[j] = carry + incl - v;\n      carry += __shfl_sync(0xffffffffu, incl, 31);\n    }\n    if (tid == 0) S.jrow[nj] = carry;\n  }\n  CTA_SYNC();\n  // phase A\'s x into xA (this CTA\'s share of the rows), before the producers start: its x reads do not queue behind\n  // every CTA\'s first weight stages\n  wgs_gather(p, S);\n  DBG(7);\n  WS_JOIN();\n  DBG(6);\n  {\n    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n    int seq = 0;\n    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    DBG(2);\n    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n  }\n#else\n  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n#endif\n#if OKC_WS\n  WS_JOIN();  // the helper\'s last epilogue stores are visible\n#endif\n  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still\n  // waits for this whole grid to finish and its writes to be visible)\n  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n  // grid barrier on this launch\'s counter (two, by generation: the other one was zeroed by this launch for the next);\n  // the last arriver bumps the generation, which no CTA of this launch reads again\n#if OKC_FINPF && OKC_FINCOL && !OKC_WGS\n  // the CTA\'s first final item: its routing (pos) and route weights are inputs, so they load before the barrier\n  constexpr int FSEGP = HD / (NT * 4);\n  const int itp = blockIdx.x;\n  const bool hasp = itp < p.M * FSEGP;\n  const int tkp = hasp ? itp / FSEGP : 0;\n  int qvp[8];\n  float wvp[8];\n#pragma unroll\n  for (int s = 0; s < 8; ++s) {\n    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;\n    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;\n  }\n#endif\n  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);\n    __threadfence();\n    const int old = atomicAdd(bc, 1);\n    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);\n    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);\n  }\n  CTA_SYNC();\n  DBG(4);\n  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA\'s share)\n  {\n    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;\n    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters\n    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;\n#if OKC_WGS\n    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;\n    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;\n#endif\n  }\n  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n#if OKC_WGS && OKC_WGSFINSS\n  // item = one token\'s whole row (thread = 8 columns): pass 1 sums each local route\'s y^2 over the row (warp\n  // shuffles, then the 16 warps\' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)\n  {\n    static_assert(HD == NT * 8, "8 columns per thread");\n    float* red = reinterpret_cast<float*>(xbuf);\n    const int lane = tid & 31, warp = WARP_IDX(tid);\n    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n      int qv[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n      float ps[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        ps[sl] = 0.f;\n        if (qv[sl] >= 0) {\n          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;\n        }\n      }\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl)\n        if (qv[sl] >= 0) {  // uniform over the CTA (one token)\n#pragma unroll\n          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);\n        }\n      if (lane == 0) {\n#pragma unroll\n        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];\n      }\n      CTA_SYNC();\n      if (tid < 8) {\n        float tsum = 0.f;\n        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];\n        red[NW * 8 + tid] = tsum;\n      }\n      CTA_SYNC();\n      float acc[8];\n#pragma unroll\n      for (int i = 0; i < 8; ++i) acc[i] = 0.f;\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl)\n        if (qv[sl] >= 0) {\n          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;\n          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;\n        }\n      uint4 o;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);\n#pragma unroll\n      for (int i = 0; i < 4; ++i) {\n        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);\n        ow[i] = *reinterpret_cast<uint32_t*>(&v);\n      }\n      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n      CTA_SYNC();  // red is reused by the next token\n    }\n  }\n#elif OKC_FINCOL\n  // column-parallel: item = (token, 1024-column segment); thread = 4 columns\n  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)\n  int it0 = blockIdx.x;\n#if OKC_FINPF\n  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math\n    const int col = (itp % FSEG) * (NT * 4) + tid * 4;\n    float ssv[8];\n    float4 yv[8];\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;\n      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);\n    }\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      if (qvp[s] >= 0) {\n        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);\n        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;\n  }\n  it0 += gridDim.x;\n#endif\n  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {\n    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));\n        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;\n  }\n#else\n  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n    float acc[16];\n#pragma unroll\n    for (int i = 0; i < 16; ++i) acc[i] = 0.f;\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;\n#pragma unroll\n        for (int i = 0; i < 4; ++i) {\n          float4 v = __ldcg(src + i);\n          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;\n        }\n      }\n    }\n    uint4 o[2];\n    uint32_t* ow = reinterpret_cast<uint32_t*>(o);\n#pragma unroll\n    for (int i = 0; i < 8; ++i) {\n      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);\n      ow[i] = *reinterpret_cast<uint32_t*>(&v);\n    }\n    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);\n    dst[0] = o[0]; dst[1] = o[1];\n  }\n#endif\n  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered\n}\n\n}  // namespace okc\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\nstatic size_t okc_smem_bytes() {\n  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;\n#if OKC_WGS\n  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;\n#endif\n  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);\n  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;\n  return base + (xa > xb ? xa : xb);\n}\n\nstatic int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n\nvoid okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {\n  okc::Params p;\n  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();\n  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();\n  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();\n  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();\n  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();\n  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }\n  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)\n#if OKC_WGS\n  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");\n  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,\n              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");\n  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();\n#else\n  p.wAs = nullptr; p.wBs = nullptr;\n#endif\n  p.zero = 0;\n  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();\n  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;\n#if OKC_WGS && OKC_RLOAD\n  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");\n#endif\n  int* ib = (int*)scratch[0].data_ptr();\n  p.ctr = ib; p.doneA = ib + 8;\n  p.ss = (float*)scratch[1].data_ptr();\n  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");\n  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);\n  p.ybuf = p.ss + okc::EMAX;\n  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n#if OKC_WGS\n  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,\n              "OKC_WGS needs xA / xB after inter (scratch[2])");\n  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;\n  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;\n  p.xready = ib + 8 + 512;\n  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;\n#else\n  p.xA = p.xB = nullptr;\n  p.xready = nullptr;\n  p.doneAs = nullptr;\n#endif\n  p.out = (__nv_bfloat16*)out.data_ptr();\n  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n  static bool attr = false;\n  size_t sm = okc_smem_bytes();\n  if (!attr) {\n    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    (void)e0; (void)e1;\n    attr = true;\n  }\n  cudaStream_t st = at::cuda::getCurrentCUDAStream();\n  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which\n  // leaves the skinny loop\'s code and registers as tuned\n  if (g_okc_pdl) {\n    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = dim3((unsigned)grid);\n    cfg.blockDim = dim3(okc::NTL);\n    cfg.dynamicSmemBytes = sm;\n    cfg.stream = st;\n    cudaLaunchAttribute at[1];\n    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    at[0].val.programmaticStreamSerializationAllowed = 1;\n    cfg.attrs = at;\n    cfg.numAttrs = 1;\n    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);\n    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);\n    return;\n  }\n  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);\n  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);\n}\n\n\n'
MOE_MID_DEFINES += ("@moe_helper0",)

# ---- DF: data-flag A->B hand-off in the MID MoE build (harness/make_dl_df.py; moe-small E173)
def _df_patch(c):
    def s1(s, old, new):
        assert s.count(old) == 1, (old[:80], s.count(old))
        return s.replace(old, new)
    c = s1(c, "#define OKC_NOLOAD 0\n", """#define OKC_NOLOAD 0
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#define OKC_DF OKC_PRMTSC   // data-flag A->B hand-off (needs the PRMT scale decode builds)
#define INTER_SENT_ROWS 128
""")
    c = s1(c, "constexpr int X1P = 512 + 8;", """// DF: phase-B x rows (the job's inter) straight from L2 until every value is written (sentinel 0xFFFF = bf16 NaN)
__device__ __forceinline__ uint32_t df_bad(const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) b |= (uint32_t)(((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu));
  return b;
}
__device__ __forceinline__ void load_x_inter_df(const Params& p, const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  const bool act = g < cnt;
  const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + (act ? g : 0)) * ID + warp * (64 * NCH) + 16 * t);
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      x[2 * c] = act ? __ldcg(src + c * 8) : make_uint4(0, 0, 0, 0);
      x[2 * c + 1] = act ? __ldcg(src + c * 8 + 1) : make_uint4(0, 0, 0, 0);
      bad |= df_bad(x[2 * c]) | df_bad(x[2 * c + 1]);
    }
    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;
    __nanosleep(32);
  }
}

constexpr int X1P = 512 + 8;""")
    a = c.index("__device__ void skinny_ring_ws(const Params& p")
    b = c.index("__device__ void ws_helper(")
    seg = c[a:b]
    seg = s1(seg, """      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);""", """      if (PB && OKC_DF) {
        load_x_inter_df(p, S, j, lane, warp, x0);
      } else {
        if (PB) {
          if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
          __syncwarp();
        }
        load_x_regs<PB>(p, S, j, lane, warp, x0);
      }""")
    c = c[:a] + seg + c[b:]
    c = s1(c, "if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters",
           "if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters")
    c = s1(c, "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n", """#if !OKC_WGS
  if (OKC_DF) {  // every CTA is past phase B: inter is dead until the next launch; give it back its "not yet written" sentinel
    const int rows = min(INTER_SENT_ROWS, p.M * 8);
    uint4* ip = reinterpret_cast<uint4*>(p.inter);
    for (int i = blockIdx.x * NT + tid; i < rows * (ID / 8); i += gridDim.x * NT) ip[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
""")
    c = s1(c, "  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n", """  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_DF
  p.inter += (size_t)128 * 4096;   // DF sentinel rows [128, 256) of the scratch: only DF launches (MID) use them
#endif
""")
    return c


_MA_ALT_SRCS["moe_helper0df"] = _df_patch(_MA_ALT_SRCS["moe_helper0"])
MOE_MID_DEFINES = tuple("@moe_helper0df" if d == "@moe_helper0" else d for d in MOE_MID_DEFINES)


def _compose_times(fns, reps=16):
    # All candidates see the same synthetic inputs and read-only cold-cache sweep.
    graphs, outputs = [], []
    for fn in fns:
        fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out=fn()
        graphs.append(g);outputs.append(out)
    flush=torch.ones((65536,1024),device='cuda',dtype=torch.float32)
    sink=torch.empty(1024,device='cuda',dtype=torch.float32)
    events=[]
    for r in range(reps):
        order=range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))
        for i in order:
            torch.sum(flush,dim=0,out=sink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[i].replay();b.record()
            events.append((i,a,b))
    torch.cuda.synchronize()
    samples=[[] for _ in graphs]
    for i,a,b in events:samples[i].append(a.elapsed_time(b)*1000)
    return [sorted(s)[len(s)//2] for s in samples]



_stack_unpinned_prepare = prepare

def prepare(weights):
    st = _stack_unpinned_prepare(weights)
    bfp = st.get("bfp")
    if bfp is not None:
        for T in (1, 8, 16, 32):
            for name, layer in _BFP_LAYER.items():
                bfp["choice"][(layer, T)] = bfp__static_choice(layer, T, bfp)
            for name in ("qkv", "o", "un"):
                st["mm_pick"][(T, name)] = "bfp"
            can_fuse = (bfp["choice"][("pooler", T)][0] == "sk2"
                        and bfp["choice"][("router", T)][0] == "sk2")
            st["mm_pick"][(T, "pr")] = "bfpf" if T <= 32 and can_fuse else "bfp"
    # The graft's fixed Lt plans handle these shapes. Explicit pr=bfp would
    # bypass the plan and run two separate projections (measured +10..12us).
    # qkv must be True for the graft's direct Lt/attention path.
    for T in (128, 256):
        for name in ("qkv", "o", "pr", "un"):
            st["mm_pick"][(T, name)] = True
    return st

_policy_parent_prepare=prepare


def _policy_variant(st,T,mode):
    ss=dict(st)
    ss['mm_pick']=dict(st['mm_pick'])
    ss['graft_lt']=dict(st.get('graft_lt',{}))
    ss['compose_graft_enabled']=dict(st.get('compose_graft_enabled',{}))
    ss['compose_boundary_warps']=dict(st.get('compose_boundary_warps',{}))
    graft,bw,drops=mode
    ss['compose_graft_enabled'][T]=graft
    ss['compose_boundary_warps'][T]=bw
    for name in drops:ss['graft_lt'].pop((name,T),None)
    return ss


def prepare(weights):
    st=_policy_parent_prepare(weights)
    st['compose_base_mm']=dict(st['mm_pick'])
    st['compose_base_graft_lt']=dict(st['graft_lt'])
    st['compose_active']=set()
    st['compose_selection']=[]
    # Exactly preserve head behavior until a complete-chain measurement picks another path.
    st['compose_graft_enabled']={T:T in (1,8,128,256) for T in (1,8,16,32,128,256)}
    st['compose_boundary_warps']={}
    gen=torch.Generator(device=st['qkv'].device).manual_seed(1618033)
    for T in (1,8,16,32,128,256):
        original=st['compose_graft_enabled'][T]
        modes=[(original,16,()),(True,16,()),(False,16,()),(False,8,()),(False,4,())]
        # Keep measured large-T Lt plans; only compare complete glue paths.
        modes=list(dict.fromkeys(modes))
        data={'stream':torch.randn(T,HIDDEN,device=st['qkv'].device,generator=gen),
              'local_start':torch.zeros(1,device=st['qkv'].device,dtype=torch.int32)}
        o_in=(.1*torch.randn(T,1024,device=st['qkv'].device,generator=gen)).to(torch.bfloat16)
        # Synthetic only: route to its busiest EP range to avoid benchmarking idle experts.
        ids=_after_attention(data,st,o_in,None,T)[2]
        counts=torch.bincount(ids.flatten().long()//32,minlength=8)
        data['local_start'].fill_(32*int(counts.argmax()))
        variants=[_policy_variant(st,T,mode) for mode in modes]
        ts=_compose_times([lambda ss=ss:_after_attention(data,ss,o_in,None,T) for ss in variants],reps=32)
        best=min(range(len(ts)),key=ts.__getitem__)
        if ts[best]>=ts[0]-.4:best=0
        picked=variants[best]
        st['compose_graft_enabled'][T]=picked['compose_graft_enabled'][T]
        st['compose_boundary_warps'][T]=picked['compose_boundary_warps'][T]
        st['graft_lt']=picked['graft_lt']
        st['compose_selection'].append({'T':T,'region':'chain_policy','us':ts,'modes':modes,'choice':modes[best]})
    return st


_SL_GLUE_CUDA = r'''
// A3: the base Gluon lane mapping and floating-point operation order, across
// eight 128-thread CTAs. This source is embedded in the single-file candidate.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
  const float* alpha2;
  float* tail;
};

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  // start's div.full.f32 by 8192 + eps becomes this FFMA in ptxas SASS.
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}

template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
  // Gluon contracts each even square into its adjacent odd square, then uses
  // the balanced in-thread tree. No sequential sum or reassociation is allowed.
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* p=cluster.map_shared_rank(sm,dst);
      p[rank*warps_per_cta+warp]=sum;
    }
  }
  cluster.sync();
  // Each warp independently reads the original 32 (or 16) warp partials and
  // performs the identical descending XOR butterfly. Separate r1/r2 buffers
  // prevent a fast CTA from overwriting a partial still read by a slow CTA.
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
}

template<int NW,int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* px=cluster.map_shared_rank(sm,dst);
      volatile float* py=cluster.map_shared_rank(smy,dst);
      px[rank*warps_per_cta+warp]=sx;
      py[rank*warps_per_cta+warp]=sy;
    }
  }
  cluster.sync();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
}

template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];
  __shared__ float partials[3][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glf<V>(a.alpha2+col,a2+r*V);
    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
          // Preserve the signed zero from the post weight as well as the
          // sliced boundary's alpha*stream FMA contraction.
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
        // carry is stored after both norms are formed.
      }
    }
  }
  // DSMEM destinations must exist before the first remote store. All work
  // above this barrier and the dependency wait is independent of the branch.
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
      // carry is stored after both norms are formed.
    }
  }
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    #pragma unroll
    for(int j=0;j<V;j++) {
      int k=r*V+j;
      carry[k]=gm(a2[k],ns[k]);
      masked[k]=(col+j>=1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);
  if(rank==0 && tid==0) a.tail[row]=tail;
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}

template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// graft: boundary 1 (attention residual), BF16 [rows, 8192] branch, rows <= 16, optional FP32 router copy.
void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0),(const float*)alpha2.data_ptr(),(float*)tail.data_ptr()};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
'''

@_lru_cache(None)
def _sl_cluster_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='sl_early_tail_' + _hashlib.md5(_SL_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_SL_GLUE_CUDA, functions=['b1'],
                            extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])

@triton.jit
def _sl_cluster_finish(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0,1024)
    beta = tl.load(BETA+c)
    gp = tl.load(POST+c).to(tl.float32)
    carry = tl.load(CARRY+t*8192+c)
    tail = tl.load(TAIL+t)
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o)
    gn = tl.load(NEXT+o).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    f = (tl.load(P+t*1024+c).to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)
    n = (f*inv).to(tl.bfloat16).to(tl.float32)
    head = carry+beta*n
    invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)
    result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)
    tl.store(SOUT+t*8192+o,result)
    tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))

SL_CLUSTER_CONFIG={1: (8, 256), 8: (16, 1024)}
def _sl_cluster_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _graft_after_attention(data,state,o_in,gate,T)
    g=state['graft']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _sl_graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
    return out, s2, ids


_sl_previous_after_attention=_after_attention

def _after_attention(data,state,o_in,gate,T):
    if T in SL_CLUSTER_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
        return _sl_cluster_after_attention(data,state,o_in,gate,T)
    return _sl_previous_after_attention(data,state,o_in,gate,T)

_sl_previous_prepare=prepare

def prepare(weights):
    st=_sl_previous_prepare(weights)
    _sl_cluster_module()
    return st
@triton.jit
def _sl_graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr):
    # our _pool_route_finish structure: program (t, 0) = pre-expert norm of the pooled row, (t, 1) = top-8 pick;
    # one warp each, both after the wait. Arithmetic is the base's _route/_route_split (fp32 pooled, 1 + w in fp32).
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_ROUTE_CONFIG={1: (8, 0, 0), 8: (4, 0, 0)}
@triton.jit
def _sl_tile_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr):
    # our _pool_route_finish structure: program (t, 0) = pre-expert norm of the pooled row, (t, 1) = top-8 pick;
    # one warp each, both after the wait. Arithmetic is the base's _route/_route_split (fp32 pooled, 1 + w in fp32).
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind<PARTS:
        c=tl.arange(0,4096)
        oc=kind*(4096//PARTS)+tl.arange(0,4096//PARTS)
        if PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        _gdc_launch()
        _gdc_wait()
        if SPLIT:
            pooled=tl.load(POOLED+t*4096+c).to(tl.float32)
        else:
            pooled=tl.load(ALL+t*4608+c)
        inv=tl.rsqrt(tl.sum(pooled*pooled,0)/4096.0+1e-5)
        if not PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        if PARTS==1:
            vals=pooled
        elif SPLIT:
            vals=tl.load(POOLED+t*4096+oc).to(tl.float32)
        else:
            vals=tl.load(ALL+t*4608+oc)
        tl.store(XOUT+t*4096+oc,(vals*inv*gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        _gdc_launch()
        _gdc_wait()
        if not PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_TILE_CONFIG={1:(4,1,True),8:(4,4,True)}
def _sl_tile_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _graft_after_attention(data,state,o_in,gate,T)
    g=state['graft']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
    return out, s2, ids


_sl_tile_prior_after=_after_attention
def _after_attention(data,state,o_in,gate,T):
    if T in SL_TILE_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
        return _sl_tile_after_attention(data,state,o_in,gate,T)
    return _sl_tile_prior_after(data,state,o_in,gate,T)

# Physical helper warp zero in the small-token native builds.
MOE_TINY2_DEFINES += ("@moe_helper0",)
MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1")


# QKV <= 8 direct escape reads, preserving decode's PDL wait and launchers.
_DE_EXT=None
_DE_RUN=bfp__run_sk2
_DE_PREPARE=prepare

def _de_source():
 cfg=('qkv8n_r144','qkv_gate',2304,24,3,1,4,2,2,1)
 _,nstage=bfp__sk2_geom(cfg,8)
 dispatch=f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;'
 source=bfp_SK2_SRC.replace('@@SK2_DISPATCH@@',dispatch).replace('@@SK2F_DISPATCH@@','')
 source=source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)','4').replace('NW + (COMP ? NESC : 0)','NW')
 source=source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");','')
 a=source.index('      int pe = 1;');b=source.index('\n    }\n    mbar_arrive(cbar);',a)
 source=source[:a]+r"""
      for(int pe=1+ew;pe<=cnt;pe+=NESC) {
        const int2 e=ptab_s[pe];const int kk=e.x&65535;
        if(lane<M && kk>=gs0*KST && kk<(gs0+nst)*KST)
          cr[lane*CP+(e.x>>16)] += __int_as_float(e.y)*__bfloat162float(x[(size_t)lane*K+kk]);
      }
"""+source[b:]
 shared=bfp_CUDA_SRC.replace('@@MBAR@@',bfp_MBAR_HELPERS).replace('@@BODY@@','')
 return shared+bfp_MAP_SRC+source

def _de_ext():
 global _DE_EXT
 if _DE_EXT is None:
  import hashlib
  source=_de_source();tag=hashlib.md5(source.encode()).hexdigest()[:12];source=source.replace('sk2_kernel','sk2_kernel_narrow_'+tag).replace('launch_sk2','launch_sk2_narrow_'+tag);cpp=bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]+'void sk2_set_pdl(int64_t n);'
  _DE_EXT=load_inline('proj_dl_escape_'+hashlib.md5(source.encode()).hexdigest()[:12],'#include <torch/extension.h>\n'+cpp,
    cuda_sources=source,functions=['sk2','sk2_set_pdl'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
 return _DE_EXT

def prepare(weights):
 _de_ext()
 return _DE_PREPARE(weights)

def bfp__run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='qkv_gate' or not 2<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)
 name=bfp__sk2_name(layer,m);ci,stream,tab,pmax,b4,n_rows=state['sk2'][name]
 out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 ext=_de_ext();ext.sk2_set_pdl(bfp_SK2_PDL)
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
 return out

# Fresh-thread primary-context binding, following matvey's helper and the
# projection team's verified entry wrapper f3721f59bc823f261742737adb2778a7.
# GPU68 reproduced invalid-context 201 in the decode QKV TMA path before this fix.
import threading as _GlueContextThreading
_GLUE_CONTEXT_TLS = _GlueContextThreading.local()
_GLUE_CONTEXT_API = None

def _glue_context_api():
    global _GLUE_CONTEXT_API
    if _GLUE_CONTEXT_API is None:
        api = _C.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_C.POINTER(_C.c_void_p)]),
            ("cuCtxGetDevice", [_C.POINTER(_C.c_int)]),
            ("cuDeviceGet", [_C.POINTER(_C.c_int), _C.c_int]),
            ("cuDevicePrimaryCtxRetain", [_C.POINTER(_C.c_void_p), _C.c_int]),
            ("cuCtxSetCurrent", [_C.c_void_p]),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, _C.c_int
        _GLUE_CONTEXT_API = api
    return _GLUE_CONTEXT_API

def _glue_context_check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")

def _glue_bind_context(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_GLUE_CONTEXT_TLS, "device", None) == index:
        return
    api = _glue_context_api()
    context = _C.c_void_p()
    _glue_context_check(api.cuCtxGetCurrent(_C.byref(context)), "cuCtxGetCurrent")
    current_device = _C.c_int(-1)
    if context.value:
        _glue_context_check(api.cuCtxGetDevice(_C.byref(current_device)), "cuCtxGetDevice")
    if not context.value or current_device.value != index:
        device, primary = _C.c_int(), _C.c_void_p()
        _glue_context_check(api.cuDeviceGet(_C.byref(device), index), "cuDeviceGet")
        _glue_context_check(api.cuDevicePrimaryCtxRetain(_C.byref(primary), device), "cuDevicePrimaryCtxRetain")
        _glue_context_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _GLUE_CONTEXT_TLS.device = index

_glue_context_parent_kernel = kernel_fn
def kernel_fn(data, state):
    _glue_bind_context(data["attn_input"])
    return _glue_context_parent_kernel(data, state)



@triton.jit
def _router_input_norm(S1, G, OUT):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    s1 = tl.load(S1 + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(OUT + t * 8192 + offs, hh)


def _pool_router_fp32(h, hr, T, state, ppick):
    if T <= 8:
        if state.get("bfp") is not None and ppick in ("bfp", "bfpf"):
            if ppick == "bfpf":
                _bfp_pdl(T, "pool")
            pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        else:
            pooled = F.linear(h, state["pr"][:D_EXPERT])
        logits = _router_fp32_gemv(hr, state)
        return (pooled, logits), -1
    if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
        _bfp_pdl(T, "pool")
        return bfp_fused_pooler_router(h, state["bfp"]), -1
    if ppick == "bfp" and state.get("bfp") is not None:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)
        return (pooled, logits), -1
    return dense(h, state["pr"], T, state, "pr")


_tune_with_bias_restore = _tune


def _tune(st, log=False):
    bias0 = st["bias"].clone()
    try:
        return _tune_with_bias_restore(st, log=log)
    finally:
        st["bias"].copy_(bias0)


_moe_global_cache = _moe


class _RouterExt:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __deepcopy__(self, memo):
        return self


def _moe(M=None, state=None):
    if state is None:
        return _moe_global_cache(M)
    sfx = tuple(state["ma_sfx"])
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        key, defines = "tiny", MOE_TINY_DEFINES
    elif M is not None and MOE_TINY2_DEFINES and M <= MOE_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES)):
        key, defines = "tiny2", MOE_TINY2_DEFINES
    elif M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        key, defines = "mid", MOE_MID_DEFINES
    elif M is not None and 16 < M <= 32 and tuple(_main_defines(sfx)) == tuple(MOE_DEFINES):
        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)
    else:
        key, defines = "main", _main_defines(sfx)
    ext = state["ma_ext"].get(key)
    if ext is None:
        ext = _RouterExt(_ma_ext(tuple(defines) + sfx))
        if MOE_HAS_PDL:
            ext.okc_set_pdl(0)
        state["ma_ext"][key] = ext
    return ext


_router_fp32_parent_prepare = prepare


def prepare(weights):
    st = _router_fp32_parent_prepare(weights)
    for T in (1, 8, 16, 32, 128, 256):
        _moe(T, st)
    return st



_ROUTER_FP32_CUDA = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__global__ __launch_bounds__(256) void router_fp32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int m) {
  const int expert = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = w + (size_t)expert * 8192;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int k = tid * 4; k < 8192; k += 1024) {
    const float4 b = *(const float4*)(row + k);
#pragma unroll
    for (int t = 0; t < 8; ++t) if (t < m) {
      const float4 a = *(const float4*)(x + (size_t)t * 8192 + k);
      acc[t] = fmaf(a.x, b.x, acc[t]);
      acc[t] = fmaf(a.y, b.y, acc[t]);
      acc[t] = fmaf(a.z, b.z, acc[t]);
      acc[t] = fmaf(a.w, b.w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int d = 16; d; d >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], d);
  }
  __shared__ float partial[8][8];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) partial[warp][t] = acc[t];
  }
  __syncthreads();
  if (warp < m) {
    float v = lane < 8 ? partial[lane][warp] : 0.f;
#pragma unroll
    for (int d = 4; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0) out[(size_t)warp * 256 + expert] = v;
  }
}

int router_fp32_launch(int64_t x, int64_t w, int64_t out, int m, int64_t stream) {
  if (m < 1 || m > 8) return -1;
  router_fp32_kernel<<<256, 256, 0, (cudaStream_t)stream>>>(
      (const float*)x, (const float*)w, (float*)out, m);
  return (int)cudaGetLastError();
}
"""
_ROUTER_FP32_EXT = None


def _router_fp32_ext():
    global _ROUTER_FP32_EXT
    if _ROUTER_FP32_EXT is None:
        import hashlib
        digest = hashlib.md5(_ROUTER_FP32_CUDA.encode()).hexdigest()[:12]
        cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint router_fp32_launch(int64_t,int64_t,int64_t,int,int64_t);'
        _ROUTER_FP32_EXT = load_inline(
            name='router_fp32_' + digest,
            cpp_sources=cpp,
            cuda_sources=_ROUTER_FP32_CUDA,
            functions=['router_fp32_launch'],
            no_implicit_headers=True,
            with_pytorch_error_handling=False,
            verbose=False,
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'])
    return _ROUTER_FP32_EXT


def _router_fp32_gemv(x, state):
    m = x.shape[0]
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    rc = state['router_fp32_ext'].router_fp32_launch(
        x.data_ptr(), state['router_fp32'].data_ptr(), out.data_ptr(), m,
        torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'FP32 router launch failed: {rc}')
    return out


_router_fp32_native_parent_prepare = prepare


def prepare(weights):
    state = _router_fp32_native_parent_prepare(weights)
    state['router_fp32_ext'] = _RouterExt(_router_fp32_ext())
    return state


# ---- local stand-in for glue_kimi's prepare-order fix (0389646a sets state['router_fp32_ext'] after the parent prepare,
# whose compose tuning already runs the T <= 8 FP32 router): bind the module on first use. Replaced by kimi's fixed hash.
_kfix_router_fp32_gemv = _router_fp32_gemv
def _router_fp32_gemv(x, state):
    if state.get('router_fp32_ext') is None:
        state['router_fp32_ext'] = _RouterExt(_router_fp32_ext())
    return _kfix_router_fp32_gemv(x, state)


# ---- opus_glue (design A) on the T <= 8 FP32 router (0389646a): boundary 1 by deferred normalization at T = 1.
# The pooler and router are linear in h = r2 * pre * ns, ns = alpha * s + beta * r1 * P (the reference's FP32 boundary on
# the BF16 o output P; SGLang feeds its router this h in FP32 at T <= 8). o's epilogue forms two FP32 rows per token,
# u = alpha * s * pre and v = beta * P * pre (pre = 1 + w in FP32), and writes them twice: FP32 for the router (0389's
# FP32 GEMV: FP32 x, FP32 weight, FP32 reduction) and BF16 for the pooler; plus per-CTA partial sums of P^2 and of the
# ns / carry squares (its input-side vectors are copied to shared memory before its griddepcontrol.wait). The route forms
# r1 = rsqrt(mean(P^2) + 1e-8) and r2 from the partials before its wait and combines pooled = r2 * (Wu + r1 * Wv),
# logits = r2 * (Ru + r1 * Rv) in FP32 after the GEMVs. Extra route programs that do not wait form the residual carry
# alpha2 * ns (FP32, b1's arithmetic); the tail sum of the carry squares comes from the partials. No b1 launch.
GLE_T = {1: True}   # T = 8 keeps 0389's b1
GLE_FRT = True      # T <= 8: the fused pooler + FP32 router kernel (else 0389's split pooler / FP32 GEMV)
GLE_ILV = 1         # its 4 FP32 router stages interleave the pooler stream: 1 last slot of each period, 2 first slot after the PDL pre-issue, 0 off
GLE_CAPTURE = None   # diagnostics: a dict receives this call's pooler / router outputs and partials
GLE_RC = {1: 8}   # T -> NCH: NCH * T extra route programs (no wait) form the carry, 8192 / (NCH * T) columns x T tokens each
_GLE_EXT = None


def _gle_source():
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == bfp_SK2_PICK[("o", 8)])
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    assert (layer, R, NWR * WSK, CL) == ("o", 32, 4, 1), cfg
    _, nstage = bfp__sk2_geom(cfg, 8)
    s = bfp_SK2_SRC
    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\n"
                  "struct GleEpi { const float *s, *a, *b, *a2; const __nv_bfloat16* p; __nv_bfloat16* uv; float* part; float* uvf; const float *x32, *w32; int fpr, ilv; };\n"
                  "static GleEpi g_gle;", 1)
    s = s.replace("           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",
                  "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {", 1)
    # the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) load before the wait
    pre = r"""  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  // the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) go to shared memory by
  // cp.async before the wait and are awaited only in the epilogue: no registers held across the main loop (2 CTAs / SM
  // next to attention) and no stall behind the pre-wait weight stages
  // ep.s is data['stream'], the layer's input residual stream: nothing in the call writes it, which is what makes this
  // pre-wait copy safe (keep it that way: any in-call writer of data['stream'] must move this copy after the wait)
  if (!FUSE && tid < 8 * M) {   // o only (this source also holds the fused pooler)
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&gx_s[tid]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d0), "l"(ep.s + (size_t)(tid >> 3) * N + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])), "l"(ep.a + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])), "l"(ep.b + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])), "l"(ep.a2 + col) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])), "l"(ep.p + col) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
"""
    anchor = '  __syncthreads();\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread'
    assert s.count(anchor) == 1
    s = s.replace(anchor, pre + anchor, 1)
    epi = r"""  constexpr int R4 = R / 4;
  if constexpr (!FUSE) {   // o: the gle epilogue; the fused pooler keeps the stock epilogue loop below
  static_assert(R4 == 8 && CL == 1 && sizeof(OutT) == 2, "gle epilogue: o8h geometry, BF16 out");
    const int tt = tid >> 3, row = (tid & 7) * 4;
    const bool on = tid < R4 * M;
    float q[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (on) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (COMP) {
#pragma unroll
        for (int c = 0; c < NESC; ++c) {
          v.x += corr[c * CSZ + tt * CP + row];
          v.y += corr[c * CSZ + tt * CP + row + 1];
          v.z += corr[c * CSZ + tt * CP + row + 2];
          v.w += corr[c * CSZ + tt * CP + row + 3];
        }
      }
#pragma unroll
      for (int w = 0; w < WSK; ++w) {
        const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
      }
      const size_t col = (size_t)cta * R + row;
      OutT* o = out + (size_t)tt * N + col;
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
      const float pf[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
      asm volatile("cp.async.wait_all;" ::: "memory");   // this thread's own copies
      const uint2 xp = gx_p[tid];
      const float4 xs = gx_s[tid], xa = gx_a[tid], xb = gx_b[tid], xa2 = gx_a2[tid];
      const float4 xas = make_float4(__fmul_rn(xa.x, xs.x), __fmul_rn(xa.y, xs.y), __fmul_rn(xa.z, xs.z), __fmul_rn(xa.w, xs.w));
      // pre = 1 + pre_ffn_norm in FP32 (the reference's (1 + w), not a BF16-rounded copy)
      const float pp[4] = {__fadd_rn(1.f, __uint_as_float(xp.x << 16)), __fadd_rn(1.f, __uint_as_float(xp.x & 0xffff0000u)),
                           __fadd_rn(1.f, __uint_as_float(xp.y << 16)), __fadd_rn(1.f, __uint_as_float(xp.y & 0xffff0000u))};
      const float ex_as[4] = {xas.x, xas.y, xas.z, xas.w}, ex_b[4] = {xb.x, xb.y, xb.z, xb.w};
      const float ex_a2[4] = {xa2.x, xa2.y, xa2.z, xa2.w};
      float uu[4], vv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float as = ex_as[j], bp = __fmul_rn(ex_b[j], pf[j]);
        uu[j] = __fmul_rn(as, pp[j]);
        vv[j] = __fmul_rn(bp, pp[j]);
        q[0] = __fmaf_rn(pf[j], pf[j], q[0]);
        q[1] = __fmaf_rn(as, as, q[1]);
        q[2] = __fmaf_rn(as, bp, q[2]);
        q[3] = __fmaf_rn(bp, bp, q[3]);
        if (col >= 1024) {
          const float x = __fmul_rn(ex_a2[j], as), y = __fmul_rn(ex_a2[j], bp);
          q[4] = __fmaf_rn(x, x, q[4]);
          q[5] = __fmaf_rn(x, y, q[5]);
          q[6] = __fmaf_rn(y, y, q[6]);
        }
      }
      // u, v: FP32 rows for the router (uvf) and BF16 rows for the pooler (uv); rows tt (u) and M + tt (v)
      auto put = [&](const float* f, size_t r) {
        __nv_bfloat162 h0 = __floats2bfloat162_rn(f[0], f[1]), h1 = __floats2bfloat162_rn(f[2], f[3]);
        uint2 hk;
        hk.x = *reinterpret_cast<uint32_t*>(&h0); hk.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(ep.uv + r * N + col) = hk;
        *reinterpret_cast<float4*>(ep.uvf + r * N + col) = make_float4(f[0], f[1], f[2], f[3]);
      };
      put(uu, (size_t)tt);
      put(vv, (size_t)(M + tt));
    }
    // the 8 lanes of one token (one aligned lane group) sum their columns: one partial row per (CTA, token)
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 4);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 2);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 1);
    }
    if (on && (tid & 7) == 0) {
      float4* pq = reinterpret_cast<float4*>(ep.part + ((size_t)cta * M + tt) * 8);
      pq[0] = make_float4(q[0], q[1], q[2], q[3]);
      pq[1] = make_float4(q[4], q[5], q[6], 0.f);
    }
  }
  else
  for (int idx = tid; idx < R4 * M; idx += NC) {
"""
    anchor = '  constexpr int R4 = R / 4;\n  for (int idx = tid; idx < R4 * M; idx += NC) {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, epi, 1)
    # FUSE with ep.fpr: the router piece is an FP32 GEMV on CUDA cores (FP32 x, FP32 weight, FP32 FMAs and reductions)
    # through the same ring as the BF16 hi + lo router: 4 stages per CTA, each = 8 router rows x 512 columns of FP32
    # weights (16 KB, 8 bulk copies) + M FP32 x rows x 512 columns (bulk copies), consumed from shared memory
    fpr_stage = (
        "        } else if (ep.fpr) {   // FP32 router stage j = st - nst: columns rq * 2048 + 512 j .. + 511\n"
        "          const int kb = rq * 2048 + (st - nst) * 512;\n"
        "          mbar_expect_tx(full0 + 8 * slot, 8 * 2048 + M * 2048);\n"
        "          for (int r = 0; r < 8; ++r)\n"
        "            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);\n"
        "          for (int t = 0; t < M; ++t)\n"
        "            bulk_g2s(s0 + 16384 + t * 2048, reinterpret_cast<const uint8_t*>(ep.x32 + (size_t)t * 8192 + kb), 2048, full0 + 8 * slot);\n"
        "        } else {   // router piece: global router stage rq * 4 + j of tile rt")
    fpr_cons = (
        "  if constexpr (FUSE) {\n    if (ep.fpr) {\n"
        "      static_assert(!FUSE || (NW == 8 && SST >= 32768), \"fp32 router stages: 8 consumer warps, 32 KB slots\");\n"
        "      // thread: column group cg = tid & 127 (4 of the stage's 512 columns) x all 8 router rows x tokens 4h .. 4h + 3\n"
        "      // (h = tid >> 7): every shared-memory float4 is read by 2 threads, not 8 warps; FP32 FMAs\n"
        "      const int cg = tid & 127, th = tid >> 7;\n"
        "      float af[32];\n"
        "#pragma unroll\n"
        "      for (int i = 0; i < 32; ++i) af[i] = 0.f;\n"
        "      for (int st = nst; st < nstt; ++st) {\n"
        "        const int slot = st % NSTAGE;\n"
        "        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
        "        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + cg * 4;\n"
        "        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384) + cg * 4;\n"
        "        float4 xa[4];\n"
        "#pragma unroll\n"
        "        for (int u = 0; u < 4; ++u)\n"
        "          xa[u] = (th * 4 + u < M) ? *reinterpret_cast<const float4*>(xs + (th * 4 + u) * 512) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
        "#pragma unroll\n"
        "        for (int r = 0; r < 8; ++r) {\n"
        "          const float4 bw = *reinterpret_cast<const float4*>(ws + r * 512);\n"
        "#pragma unroll\n"
        "          for (int u = 0; u < 4; ++u) {\n"
        "            float v = af[r * 4 + u];\n"
        "            v = fmaf(xa[u].x, bw.x, v); v = fmaf(xa[u].y, bw.y, v); v = fmaf(xa[u].z, bw.z, v); v = fmaf(xa[u].w, bw.w, v);\n"
        "            af[r * 4 + u] = v;\n"
        "          }\n"
        "        }\n"
        "        __syncwarp();\n"
        "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
        "      }\n"
        "      // reduction over the 128 column groups through the (now idle) ring: red[th][cg][33] (static indices only)\n"
        "      if constexpr (COMP) mbar_wait(cbar, 0);   // as the BF16 router: the escape warps are done with the ring\n"
        "      named_bar(1, NC);                         // and every consumer is done reading its last stage\n"
        "      float* red32 = reinterpret_cast<float*>(gbase);\n"
        "#pragma unroll\n"
        "      for (int i = 0; i < 32; ++i) red32[(th * 128 + cg) * 33 + i] = af[i];\n"
        "      named_bar(1, NC);\n"
        "      float* red2 = red32 + 256 * 33;\n"
        "      {\n"
        "        const int o = tid & 63, pq = tid >> 6, h = o >> 5, e = o & 31;\n"
        "        float v = 0.f;\n"
        "        for (int c = pq * 32; c < pq * 32 + 32; ++c) v += red32[(h * 128 + c) * 33 + e];\n"
        "        red2[pq * 64 + o] = v;\n"
        "      }\n"
        "      named_bar(1, NC);\n"
        "      if (tid < 64) {\n"
        "        const int h = tid >> 5, e = tid & 31, r = e >> 2, t = h * 4 + (e & 3);\n"
        "        const float v = red2[tid] + red2[64 + tid] + red2[128 + tid] + red2[192 + tid];\n"
        "        if (t < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + t * 8 + r, v);\n"
        "      }\n"
        "      named_bar(1, NC);\n"
        "    } else {\n    // router piece:")
    # ep.ilv: the 4 FP32 router stages sit at ring positions PER - 1, 2 PER - 1, ... (PER = nstt / 4) between the pooler
    # stages, so their loads and FMAs overlap the pooler stream; ring position q holds pooler stage q - q / PER otherwise
    ilv_edits = (
        ("  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n",
         "  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n"
         "  const bool ilv = FUSE && ep.fpr && ep.ilv && (nstt % 4) == 0 && nstt / 4 > P;\n"
         "  const int PER = ilv ? nstt / 4 : (1 << 30);\n"
         "  const int OFF = ilv ? (ep.ilv == 2 ? P : PER - 1) : 0;   // router stage slot inside each period\n"),
        ("        if (!FUSE || st < nst) {\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n",
         "        const bool isr = ilv ? (st % PER == OFF) : (FUSE && st >= nst);\n"
         "        const int ps = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
         "        if (!isr) {\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + ps) * wstride, WBY, full0 + 8 * slot);\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + ps) * UPS);\n"),
        ("          const int kb = rq * 2048 + (st - nst) * 512;\n",
         "          const int kb = rq * 2048 + (ilv ? st / PER : st - nst) * 512;\n"),
        ("        if (FUSE && st >= nst) {   // router stages:",
         "        if (ilv ? (st % PER == OFF) : (FUSE && st >= nst)) {   // router stages:"),
        ("        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;\n",
         "        const int psq = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
         "        const int kbeg = (gs0 + psq) * KST, kend = kbeg + KST;\n"),
        ("        if ((st % NESC) != ew) {", "        if ((psq % NESC) != ew) {"),
        ("  for (int st = 0; st < nst; ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n",
         "  float afi[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   // ilv: FP32 router accumulators (warp = router row)\n"
         "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
         "    if constexpr (FUSE) if (ilv && st % PER == OFF) {   // an interleaved FP32 router stage\n"
         "      const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + warp * 512;\n"
         "      const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384);\n"
         "#pragma unroll\n"
         "      for (int i = 0; i < 4; ++i) {\n"
         "        const int k = lane * 4 + i * 128;\n"
         "        const float4 bw = *reinterpret_cast<const float4*>(ws + k);\n"
         "#pragma unroll\n"
         "        for (int t = 0; t < 8; ++t) if (t < M) {\n"
         "          const float4 a = *reinterpret_cast<const float4*>(xs + t * 512 + k);\n"
         "          afi[t] = fmaf(a.x, bw.x, afi[t]); afi[t] = fmaf(a.y, bw.y, afi[t]);\n"
         "          afi[t] = fmaf(a.z, bw.z, afi[t]); afi[t] = fmaf(a.w, bw.w, afi[t]);\n"
         "        }\n"
         "      }\n"
         "      __syncwarp();\n"
         "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
         "      continue;\n"
         "    }\n"),
        ("  if constexpr (FUSE) {\n    if (ep.fpr) {\n",
         "  if constexpr (FUSE) {\n    if (ilv) {   // interleaved FP32 router: warp butterflies of afi, then the 4-quarter reduction below\n"
         "#pragma unroll\n"
         "      for (int t = 0; t < 8; ++t)\n"
         "#pragma unroll\n"
         "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\n"
         "      float mine = afi[0];\n"
         "#pragma unroll\n"
         "      for (int t = 1; t < 8; ++t) mine = lane == t ? afi[t] : mine;\n"
         "      if (lane < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + lane * 8 + warp, mine);\n"
         "      if constexpr (COMP) mbar_wait(cbar, 0);\n"
         "      named_bar(1, NC);\n"
         "    } else if (ep.fpr) {\n"))
    for old, new in (
            ("        } else {   // router piece: global router stage rq * 4 + j of tile rt", fpr_stage),
            ("  if constexpr (FUSE) {\n    // router piece:", fpr_cons),
            ("    named_bar(1, NC);\n    __shared__ int s_rlast;",
             "    named_bar(1, NC);\n    }\n    __shared__ int s_rlast;")):
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new, 1)
    for old, new in ilv_edits:
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));", 1)
    s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);", 1)
    host = (f"void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,\n"
            f"         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {{")
    assert s.count(host) == 1
    disp = (f"if (tokr == 8 && N == 8192 && K == 1024 && M <= 8) {{ launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
            f"{NACC}, {CL}, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr); return; }}")
    pcfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "pooler")
    flines = []
    for tokr in (8,):   # the fused pooler + FP32 router at M <= 8
        _, pns = bfp__sk2_geom(pcfg, tokr)
        _, _, _, pR, pNT, pNWR, pWSK, pUPW, pNACC, pCL = pcfg
        flines.append(f"if (tokr == {tokr}) {{ launch_sk2<1, {tokr}, {pNT}, {pNWR}, {pWSK}, {pUPW}, {pns}, true, false, "
                      f"{pNACC}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                      f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                      f"rcnt.data_ptr<int>()); return; }}")
    s = s.replace("@@SK2_DISPATCH@@", disp).replace("@@SK2F_DISPATCH@@", "\n  ".join(flines))
    s += r"""
void gle_o(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
           at::Tensor s, at::Tensor a, at::Tensor b, at::Tensor a2, at::Tensor p, at::Tensor uv, at::Tensor part, at::Tensor uvf) {
  TORCH_CHECK(M >= 1 && M <= 8 && s.size(0) == M && uv.size(0) == 2 * M && uvf.size(0) == 2 * M && part.numel() >= 256 * M * 8, "gle_o: shapes");
  g_gle = GleEpi{s.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), a2.data_ptr<float>(),
                 reinterpret_cast<const __nv_bfloat16*>(p.data_ptr()), reinterpret_cast<__nv_bfloat16*>(uv.data_ptr()),
                 part.data_ptr<float>(), uvf.data_ptr<float>()};
  sk2(Ws, x, out, ptab, pmax, base4, M, 8192, 1024, 0, 8, 0, ptab);
}
void gle_pr(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
            int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, at::Tensor x32, at::Tensor w32, int64_t wpf) {
  // fused pooler (BF16 x rows) + FP32 router (FP32 x32 rows, FP32 weights w32) at M <= 8
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x32.size(0) == M && x32.scalar_type() == at::kFloat &&
              w32.scalar_type() == at::kFloat && w32.size(0) == 256 && w32.size(1) == 8192, "gle_pr: shapes");
  g_gle = GleEpi{};
  g_gle.x32 = x32.data_ptr<float>(); g_gle.w32 = w32.data_ptr<float>(); g_gle.fpr = 1; g_gle.ilv = (int)wpf;
  sk2f(Ws, Wsr, x, out, outr, ptab, pmax, base4, part, rcnt, M, 8);
  g_gle = GleEpi{};
}
"""
    shared = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "")
    return shared + bfp_MAP_SRC + s


def _gle_ext():
    global _GLE_EXT
    if _GLE_EXT is None:
        source = _gle_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace("sk2_kernel", "sk2_kernel_gle" + tag).replace("launch_sk2", "launch_sk2_gle" + tag)
                  .replace("void sk2(", "void sk2_gle" + tag + "(").replace("  sk2(Ws,", "  sk2_gle" + tag + "(Ws,")
                  .replace("void sk2f(", "void sk2f_gle" + tag + "(").replace("  sk2f(Ws,", "  sk2f_gle" + tag + "(Ws,"))
        cpp = ("void gle_o(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,"
               " int64_t M, torch::Tensor s, torch::Tensor a, torch::Tensor b, torch::Tensor a2, torch::Tensor p,"
               " torch::Tensor uv, torch::Tensor part, torch::Tensor uvf);\n"
               "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
               " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
               " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\nvoid sk2_set_pdl(int64_t n);\n")
        _GLE_EXT = load_inline("gle_o_" + tag, "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
                               functions=["gle_o", "gle_pr", "sk2_set_pdl"], no_implicit_headers=True,
                               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _GLE_EXT


@triton.jit
def _gle_carry(PO, S, ALPHA, BETA, ALPHA2, t, cols, r1, ROWS1: tl.constexpr):
    # boundary 1's residual: ns = alpha * s + beta * bf16(r1 * P) (b1's FMA order), carry = alpha2 * ns
    p = tl.load(PO + t * 8192 + cols).to(tl.float32)
    s = tl.load(S + t * 8192 + cols)
    a = tl.load(ALPHA + cols)
    b = tl.load(BETA + cols)
    bn = (p * r1).to(tl.bfloat16).to(tl.float32)
    if ROWS1:
        ns = tl.fma(b, bn, a * s)
    else:
        ns = tl.where(cols < 4096, tl.fma(a, s, b * bn), tl.fma(b, bn, a * s))
    return tl.load(ALPHA2 + cols) * ns


@triton.jit
def _gle_route(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT,
               PO, S, ALPHA, BETA, ALPHA2, S1,
               MODE: tl.constexpr, PARTS: tl.constexpr, NCH: tl.constexpr, TT: tl.constexpr):
    # _sl_tile_route on the two-row (u, v) pooler / router outputs; r1 / r2 come from the o epilogue's partial sums,
    # complete before this launch (o completes before the pooler wait returns; the FP32 router is a plain launch)
    t = tl.program_id(0)
    kind = tl.program_id(1)
    k8 = tl.arange(0, 8)
    cta = tl.arange(0, 256)
    q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :]), 0)
    q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
    q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
    q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
    q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
    r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
    r2 = tl.rsqrt((q1 + 2.0 * r1 * q2 + r1 * r1 * q3) / 8192.0 + 1e-5)
    if kind > PARTS:
        # carry programs (no wait: they read o's outputs and the inputs only): column chunk (kind - PARTS - 1) * T + t
        # of 8192 // (NCH * T) columns, for every token
        cc = ((kind - PARTS - 1) * TT + t) * (8192 // (NCH * TT)) + tl.arange(0, 8192 // (NCH * TT))
        for tok in tl.static_range(TT):
            r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + tok) * 8), 0) / 8192.0 + 1e-8)
            tl.store(S1 + tok * 8192 + cc, _gle_carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1))
    elif kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        _gdc_wait()
        _gdc_launch()
        pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        else:
            vals = (tl.load(POOLED + t * 4096 + oc).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + oc).to(tl.float32) * r1) * r2
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
        q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
        q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
        tail = q4 + 2.0 * r1 * q5 + r1 * r1 * q6
        _gdc_wait()
        _gdc_launch()
        tl.store(RT + t, r1)
        tl.store(RT + T + t, tail)
        logit = (tl.load(LOGITS + t * 256 + e) + tl.load(LOGITS + (T + t) * 256 + e) * r1) * r2
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


def _gle_frt(xb, x32, state):
    """Fused BF16 pooler + FP32 router for M = xb.shape[0] <= 8 rows: (pooled bf16 [M, 4096], logits fp32 [M, 256])."""
    bst = state["bfp"]
    m = xb.shape[0]
    dev = xb.device
    if "fused_part" not in bst:
        bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((m, 256), device=dev, dtype=torch.float32)
    _bfp_pdl(m, "pool")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, logits, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
               m, x32, state["router_fp32"], int(GLE_ILV))
    return pooled, logits


_gle_prior_pool_router_fp32 = _pool_router_fp32
def _pool_router_fp32(h, hr, T, state, ppick):
    if (GLE_FRT and T <= 8 and hr is not None and ppick in ("bfp", "bfpf") and state.get("bfp") is not None
            and state.get("router_fp32") is not None and hr.is_contiguous() and h.is_contiguous()):
        return _gle_frt(h, hr, state), -1
    return _gle_prior_pool_router_fp32(h, hr, T, state, ppick)


def _gle_o_choice(state, T):
    bst = state.get("bfp")
    if bst is None or not bst.get("sm90"):
        return False
    ch = bst.get("choice", {}).get(("o", T)) or bfp__nearby_choice("o", T, bst) or bfp__static_choice("o", T, bst)
    return ch[0] == "sk2" and bfp__sk2_name("o", T) == bfp_SK2_PICK[("o", 8)]


def _gle_after_attention(data, state, o_in, gate, T):
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    g = state['graft']
    bst = state["bfp"]
    ci, stream, tab, pmax, b4, n_rows = bst["sk2"][bfp__sk2_name("o", T)]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2 * T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uvf = torch.empty((2 * T, HIDDEN), **f32)
    part = torch.empty((256 * T * 8,), **f32)
    rt = torch.empty((2, T), **f32)
    nch = GLE_RC.get(T, GLE_RC.get(str(T), 0))
    s1 = torch.empty((T, HIDDEN), **f32)
    _bfp_pdl(T, "o")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_o(stream, o_in, P, tab, pmax, b4, T, data['stream'], g['alpha1'], g['beta1'], g['alpha2'], state['g_preffn'], uv, part,
              uvf)
    if GLE_FRT:   # fused pooler + FP32 router (one launch)
        pooled, logits = _gle_frt(uv, uvf, state)
    else:         # 0389's T <= 8 split: BF16 pooler (bf16_proj sk2), FP32 router GEMV, here on the two rows per token
        pooled = _bfp_gemm(state, "pool", uv, 2 * T, D_EXPERT, HIDDEN)
        logits = _router_fp32_gemv(uvf, state)
    if GLE_CAPTURE is not None:
        GLE_CAPTURE.update(logits=logits, pooled=pooled, part=part, P=P)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _graft_al(pooled, logits, xm, ids, wts, lids, part, rt)
    _gle_route[(T, SL_TILE_CONFIG[T][1] + 1 + nch)](pooled, logits, part, T, state["bias"], state["g_preexp"], ids, wts, xm,
                                                    data["local_start"], lids, rt, P, data['stream'], g['alpha1'], g['beta1'],
                                                    g['alpha2'], s1, MODE=SL_ROUTE_CONFIG[T][1],
                                                    PARTS=SL_TILE_CONFIG[T][1], NCH=nch, TT=T,
                                                    num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mx = _moe(T, state)
    if MOE_HAS_PDL:
        mx.okc_set_pdl(1 if _pdl(T) else 0)
    for m0 in range(0, T, _MA_MMAX):
        m1 = min(T, m0 + _MA_MMAX)
        mx.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32, state["nsm"])
    U, S = dense(m, state["unpool_slice"], T, state, "un")
    if S != 1 or U.dtype != torch.bfloat16:
        raise RuntimeError('gle: BF16 sliced unpooler expected')
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    nw, bc = SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T, 8192 // bc)](U, T, s1, rt[1], g['beta2'], g['post2'], g['pre2'], s2, out, BC=bc, num_warps=nw,
                                        launch_pdl=True)
    return out, s2, ids


_gle_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    if (GLE_T.get(T, GLE_T.get(str(T), False)) and GLE_RC.get(T, 0) > 0 and gate is None and _pdl(T) and 'graft' in state
            and state.get('compose_graft_enabled', {}).get(T, True) and T in SL_TILE_CONFIG and T in SL_CLUSTER_CONFIG):
        mmp = state.get("mm_pick", {})
        ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
        opick = mmp.get((T, "o"), mmp.get((_bucket(T), "o"), False))
        moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                                   marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
        gw = state.get('g_preffn')
        if (ppick in ("bfp", "bfpf") and opick == "bfp" and moe == "agent" and state.get("bfp") is not None
                and state.get("router_fp32") is not None and state.get("router_fp32_ext") is not None
                and gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == HIDDEN and gw.is_contiguous()
                and _gle_o_choice(state, T)):
            return _gle_after_attention(data, state, o_in, gate, T)
    return _gle_prior_after(data, state, o_in, gate, T)


_gle_prior_prepare = prepare
def prepare(weights):
    _gle_ext()
    return _gle_prior_prepare(weights)


# ---- opus_glue: boundary 1's FP32 router copy as the exact FP32 activation (T <= 8 paths that keep b1).
# 0389646a's b1 writes router_input = r2 * ns * pre1 with ns built on bf16(r1 * P) and pre1 = bf16(1 + w): 2e-3 rel from
# the reference's FP32 normalized activation (router_only_test B1_HR). This copy writes
# hr = (alpha s + beta (r1 P)) * r2x * (1 + w) in FP32, r2x = rsqrt(mean(that ns^2) + 1e-5) (one more cluster sum);
# the BF16 h, the carry and the tail are unchanged. w comes in the (otherwise unused) post slot.
GLE_B1X = True
_GLE_W_CUR = None


def _gle_b1x_source():
    s = _SL_GLUE_CUDA
    for old, new in (
            ("float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];\n  __shared__ float partials[3][NW];",
             "float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N],nsx[N];\n  __shared__ float partials[4][NW];"),
            ("    if constexpr(POST) glb<V>(a.post+col,post+r*V);\n",
             "    if constexpr(POST || ROUTER) glb<V>(a.post+col,post+r*V);   // ROUTER: post = w (pre_ffn_norm)\n"),
            ("        float bn=gb(gm(b[k],r1));\n",
             "        float bn=gb(gm(b[k],r1));\n"
             "        if constexpr(ROUTER) {   // the router's FP32 copy: no BF16 rounding of r1 * P\n"
             "          float bx=gm(b[k],r1);\n"
             "          nsx[k]=(SLICED || (PRELOADED_BASE && col<4096)) ? gf(alpha[k],s[k],gm(beta[k],bx)) : gf(beta[k],bx,gm(alpha[k],s[k]));\n"
             "        }\n"),
            ("  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n",
             "  float r2, r2x=0.f;\n"
             "  if constexpr(ROUTER) {   // the router copy's sum of squares rides the same cluster exchange\n"
             "    float zs;\n"
             "    r2=gr(cluster_squares_triple<NW>(ns,masked,nsx,partials[1],partials[2],partials[3],cluster,rank,tail,zs),1.e-5f);\n"
             "    r2x=gr(zs,1.e-5f);\n"
             "  } else {\n"
             "    r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n"
             "  }\n"),
            ("template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n",
             "template<int NW,int N>\n"
             "__device__ __forceinline__ float cluster_squares_triple(const float (&x)[N], const float (&y)[N], const float (&z)[N],\n"
             "    float* sm, float* smy, float* smz, cg::cluster_group cluster, int rank, float& tail, float& zsum) {\n"
             "  constexpr int warps_per_cta=NW/8;\n"
             "  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;\n"
             "  float sx=butterfly(local_squares(x),16);\n"
             "  float sy=butterfly(local_squares(y),16);\n"
             "  float sz=butterfly(local_squares(z),16);\n"
             "  if(lane==0 && warp<warps_per_cta) {\n"
             "    #pragma unroll\n"
             "    for(int dst=0;dst<8;dst++) {\n"
             "      volatile float* px=cluster.map_shared_rank(sm,dst);\n"
             "      volatile float* py=cluster.map_shared_rank(smy,dst);\n"
             "      volatile float* pz=cluster.map_shared_rank(smz,dst);\n"
             "      px[rank*warps_per_cta+warp]=sx;\n"
             "      py[rank*warps_per_cta+warp]=sy;\n"
             "      pz[rank*warps_per_cta+warp]=sz;\n"
             "    }\n"
             "  }\n"
             "  cluster.sync();\n"
             "  tail=butterfly(smy[lane&(NW-1)],NW/2);\n"
             "  zsum=butterfly(smz[lane&(NW-1)],NW/2);\n"
             "  return butterfly(sm[lane&(NW-1)],NW/2);\n"
             "}\n\n"
             "template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n"),
            ("      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);\n",
             "      if constexpr(ROUTER) {\n"
             "        float hx[V];\n"
             "        #pragma unroll\n"
             "        for(int j=0;j<V;j++) hx[j]=gm(gm(nsx[r*V+j],r2x),ga(1.f,post[r*V+j]));\n"
             "        gsf<V>(a.router+row*8192+col,hx);\n"
             "      }\n"),
            ("void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {\n"
             "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),",
             "void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router,TV w) {\n"
             "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)w.data_ptr(),(const BF*)pre.data_ptr(),")):
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    return s


@_lru_cache(None)
def _gle_b1x_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    src = _gle_b1x_source()
    return _tvm_load_inline(name='gle_b1x_' + _hashlib.md5(src.encode()).hexdigest()[:12], cuda_sources=src,
                            functions=['b1'], extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])


class _GleB1x:
    """_sl_cluster_module() stand-in: the router-writing b1 goes to the exact-FP32 copy (w from the calling state)."""
    def __init__(self, base):
        self.base = base

    def b1(self, branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router):
        if write_router and GLE_B1X and _GLE_W_CUR is not None:
            return _gle_b1x_module().b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, True, _GLE_W_CUR)
        return self.base.b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router)


_gle_b1x_prior_module = _sl_cluster_module
def _sl_cluster_module():
    return _GleB1x(_gle_b1x_prior_module())


_gle_b1x_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    global _GLE_W_CUR
    gw = state.get('g_preffn')
    _GLE_W_CUR = gw if (gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == HIDDEN and gw.is_contiguous()) else None
    return _gle_b1x_prior_after(data, state, o_in, gate, T)


_gle_b1x_prior_prepare = prepare
def prepare(weights):
    _gle_b1x_module()
    return _gle_b1x_prior_prepare(weights)


# ---- opus_glue: pin the T <= 8 chain. 0389's compose tuner picks the T = 1 / T = 8 chain by timing inside prepare; its
# older graft chain feeds the FP32 router a BF16-rounded b1 copy (2e-3 rel). Force the chain with b1x / gle / the fused FP32
# router (exact FP32 router input) on every prepare.
_pin_parent_prepare = prepare
def prepare(weights):
    st = _pin_parent_prepare(weights)
    cge = st.setdefault('compose_graft_enabled', {})
    for T in (1, 8):
        cge[T] = True
    return st


def _make_chen():
    # (embedded: chenwainuo muse_bf16_proj)
    # Based on github.com/chenwainuo/okc muse_bf16_proj/submission.py @ 3d418e1 (Apache-2.0). Copied for the OKC starting point.
    """Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation.
v13/v14: up to 32 tokens, the BF16 projections and the router (12-bit hi + lo) run a register-direct GEMV on losslessly 12-bit-packed weights (see GVCU),
launch geometry picked per shape in prepare() by timing against the v12 path.
pw11 ("gvx"): r6-c12 unchanged (every r6 path is still baked and kept as the per-shape baseline "r6"), plus gv geometries
with 64-144 rows per CTA (x fetched once per 128 rows instead of once per 16) and a cross-CTA K split of 2/4/8, reduced
either through cluster DSMEM (path suffix a, r6's own S > 1 epilogue) or through an fp32 workspace + last-arriving CTA
(suffix b, V bit 2, no cluster); suffix -s = x in a shared-memory slab instead of through L1; the router gets the
12-bit codec on both hi and lo. Each new path is validated (one-hot x8 exact, random rows) and timed against the r6
choice in a second bake (BAKE / PICK lines); kernel_fn takes the pw11 pick when it exists, else r6's dispatch.
pw12: the same gv kernel with W streamed through a private per-warp cp.async.bulk smem ring (V bit 3, paths gb-*);
decode, mma, escapes, reduction and epilogue unchanged. PROBE gvbulk lines compare it with r6 on pooler m=1.
pw18 (= pw17 with codec E v3): r6 unchanged (its 12-bit paths, bake and dispatch are SASS-identical) plus codec E paths
(gv FMT 4, "ge-*"). The code length is fixed by a code's first 3 bits (LOP3 + PRMT on an 8-byte table), so the per-lane
bit-position chain has no memory access. The exponent pair value comes from a 256-entry smem LUT (single or lane-replicated)
off the chain. There are 1, 2 or 4 independent substreams per lane, and refill words come through a conflict-free
[substream][slot][lane] 4 B cp.async ring and are prefetched. 11.07 / 11.17 / 11.32 bits/weight (K 1 / 2 / 4) against 12
(proj_rsi/tools/CODEC_E_LAYOUT.md v3). Each pack is proven bit-exact by the reference unpacker before any E kernel runs. Each
path is validated like every second-bake path and timed against r6's pick (BAKE / PICK). PROBE ecod lines time the best E path
against r6 on pooler m=1 and m=16, alternated.
pw29: pw23 plus wgmma router paths at m = 128 / 256 (rwk, its own sm_90a extension), baked against pw23's rs / rp / rv
paths and r6's cuBLASLt path.
pw31 (scored merge): pw27h (P1: flag-zeroed split-K router m = 8/16/32 and unpooler m <= 8 on r6's gv, rr6 at router
m = 1) + pw29h (wgmma split-K router m = 128 / 256 with self-zeroing); the two regions are disjoint (3-way merge on pw23).
pw34 = pw33 + rs128 (audits/m128-plan.md): pw23 unchanged plus a swap-AB register-A split-K wgmma kernel for m exactly 128 / 256 on the
pooler, qkv_gate, unpooler and o layers: 128 W rows x all tokens per CTA, two math warpgroups decode the lossless 12-bit
pack (pw10 _pack_nat + _c12, re-blocked to 12 KB per 128-row k64 stage) into wgmma A registers, x is the smem B operand
(2D TMA, SW128), one full barrier per ring slot shared by a W bulk lane and an x TMA lane, K split over KS CTAs on a
one-wave grid, partials summed by cp.reduce.async.bulk.tensor bf16 add into out, which the kernel zeroes itself (each
CTA its slice, release counter, polled off the critical path; no memset node, no cluster). Built in a background thread
during r6's phase; BAKE / PICK lines time it against r6's own path per (layer, m); any failure leaves r6's choices."""
    from typing import TypedDict
    import math
    import statistics
    import time
    import torch
    _T0 = time.time()
    from torch.utils.cpp_extension import load_inline
    GXP = []
    GX11 = False  # pw17: pw11's wide-row gx paths never won (checks 824f32ce, f1e6a186, f03e2e77); not built
    for _ms, _xs in (((1, 8), 0), ((16, 32), 2)):
        for _wr in (9, 8):  # 144 rows: 16 x 8 = 128 CTAs; 128 rows: 2304 = 18 x 128 -> 144 CTAs
            for _ab, _v in (("a", 1), ("b", 5)):
                if _ab == "a":  # pw12 trim: q144a (cluster 8 declined on the judge, 120 < 128) and q128a dropped
                    continue
                _ms2 = tuple(m_ for m_ in _ms if not (_wr == 8 and m_ == 32))  # pw12 trim: q128b m=32 (L1-x, MT 4)
                if _ms2:
                    GXP.append(("gx-q%d%s" % (16 * _wr, _ab), "qkv_gate", _ms2, (1, _wr, 1, 16, 8, 1, _xs), _v))
    for _ab, _v in (("a", 1), ("b", 5)):
        if _ab == "b":  # pw12 trim: gx-q144a-s dropped (cluster 8 declined)
            GXP.append(("gx-q144%s-s" % _ab, "qkv_gate", (32,), (1, 9, 1, 16, 8, 1, 1), _v))
        GXP.append(("gx-p128%s" % _ab, "pooler", (16,), (1, 8, 2, 16, 4, 1, 2), _v))  # pw12 trim: m=32 L1-x (8 B spill) dropped
        GXP.append(("gx-p128%s-s" % _ab, "pooler", (32,), (1, 8, 2, 16, 4, 1, 1), _v))
        GXP.append(("gx-u128%s" % _ab, "unpooler", (16,), (1, 8, 2, 16, 2, 1, 2), _v))  # pw12 trim: m=32 L1-x dropped
        GXP.append(("gx-u128%s-s" % _ab, "unpooler", (32,), (1, 8, 2, 16, 2, 1, 1), _v))
    GXP.append(("gx-o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1))
    GXP.append(("gx-r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5))  # pw12 trim: r12c8 / r12c16 dropped
    if not GX11:
        GXP = []
    def _bv(v, ns, cu=1, fence=False):
        return v | 8 | ({2: 0, 4: 1, 8: 2, 16: 3}[ns] << 4) | ({1: 0, 2: 1, 4: 2}[cu] << 6) | (0 if fence else 256)
    GB = True
    for _nm, _ly, _ms, _g, _v, _vars in (
            ("q", "qkv_gate", (1, 8), (1, 1, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1))),
            ("q144b", "qkv_gate", (1, 8), (1, 9, 1, 16, 8, 1, 0), 5, ((8, 1),)),
            ("q", "qkv_gate", (16,), (1, 1, 4, 8, 4, 1, 2, 1), 1, ((4, 1), (8, 1))),
            ("q144b", "qkv_gate", (16, 32), (1, 9, 1, 16, 8, 1, 2), 5, ((8, 1),)),
            ("q", "qkv_gate", (32,), (1, 1, 4, 16, 2, 1, 2, 1), 1, ((4, 1), (8, 1))),
            ("o", "o", (1, 8), (1, 1, 8, 2, 1, 1, 0), 0, ((4, 1),)),
            ("o", "o", (16,), (1, 1, 8, 2, 1, 1, 0), 1, ((4, 1),)),
            ("o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1, ((4, 1),)),
            ("p", "pooler", (1, 8), (1, 2, 16, 8, 1, 1, 0), 3, ((4, 1),)),
            ("p16w", "pooler", (1, 8, 16), (1, 2, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1), (4, 2))),
            ("p", "pooler", (32,), (1, 1, 8, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
            ("p128b", "pooler", (16, 32), (1, 8, 2, 16, 4, 1, 2), 5, ((4, 1),)),
            ("u", "unpooler", (1, 8), (1, 1, 16, 4, 1, 1, 0), 1, ((4, 1),)),
            ("u", "unpooler", (16,), (1, 1, 4, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
            ("u", "unpooler", (32,), (1, 1, 4, 16, 1, 1, 2), 0, ((4, 1), (8, 1))),
            ("u128b", "unpooler", (16, 32), (1, 8, 2, 16, 2, 1, 2), 5, ((4, 1),)),
            ("r", "router", (1, 8, 16), (2, 1, 8, 2, 8, 1, 0), 0, ((4, 1),)),
            ("r", "router", (32,), (2, 1, 4, 2, 16, 1, 0), 0, ((4, 1),)),
            ("r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5, ((4, 1),))):
        for _ns, _cu in _vars if GB else ():
            if True:  # pw21: no bulk-ring paths (gb-q-n4 never won)
                continue
            GXP.append(("gb-%s-n%d%s" % (_nm, _ns, "c%d" % _cu if _cu > 1 else ""), _ly, _ms, _g, _bv(_v, _ns, _cu)))
    GE = False  # pw21: codec E closed (pw17-pw20); r6 + router m=128/256 paths only
    _ER, _EK2, _EK4 = 1 << 20, 1 << 22, 2 << 22
    _GEG = {("qkv_gate", 1): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 8): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 16): (4, 1, 8, 16, 1, 1, 0),
            ("pooler", 1): (4, 2, 8, 16, 1, 1, 0), ("pooler", 8): (4, 2, 8, 16, 1, 1, 0), ("pooler", 16): (4, 2, 8, 16, 1, 1, 0),
            ("unpooler", 1): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 8): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 16): (4, 1, 4, 16, 1, 1, 2)}
    _GEV = (("", 0), ("r", _ER), ("r-k2", _ER | _EK2), ("r-k4", _ER | _EK4))
    _GEN = {"qkv_gate": "q", "pooler": "p512", "unpooler": "u"}
    GEP = [("ge-%s%s" % (_GEN[ly] if not (ly == "unpooler" and m <= 8) else "u16", sfx), ly, (m,), g, 1 | b | (2 << 12 if ly == "qkv_gate" else 0))
           for (ly, m), g in _GEG.items() for sfx, b in _GEV if not (ly != "unpooler" and m == 16 and b & (3 << 22))] if GE else []
    _EL = 1 << 24
    GEPL = [("ge-l%s" % nm, ly, (m,), g, 1 | 2 | ((1024 // nt) << 12) | _EL)
            for nm, ly, g, nt in (("p1k", "pooler", (4, 2, 16, 8, 1, 1, 0, 1), 1024), ("p16", "pooler", (4, 1, 16, 8, 1, 1, 0, 1), 512),
                                  ("q16", "qkv_gate", (4, 1, 16, 8, 1, 1, 0, 1), 512), ("q8", "qkv_gate", (4, 1, 8, 16, 1, 1, 0, 1), 256),
                                  ("u8", "unpooler", (4, 1, 8, 8, 1, 1, 0, 1), 256))
            for m in (1, 8)] if GE else []
    GEP += GEPL
    GXP += GEP
    _ZF, _PZ, _XZ = 1 << 27, 1 << 28, 1 << 29
    GRD = []
    for _nm, _g, _ms, _vs in (("rk-s8", (2, 1, 8, 2, 8, 1, 0, 2), (8, 16, 32), "z"),           # 16 x 8 = 128 CTAs x 256 thr
                              ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16, 32), "z"),      # 256 CTAs x 256 thr, one chunk per warp
                              ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16), "x"),          # critic (pw27h): the x variant at m=32 is 272 CTAs > 264 resident slots
                              ("rk-s32", (2, 1, 4, 1, 32, 1, 0, 1), (8, 16), "x"),          # 512 CTAs x 128 thr
                              ("rk-s16w16", (2, 1, 16, 1, 8, 1, 0, 1), (8,), "z")):
        for _c in _vs:
            GRD.append((_nm + _c, "router", _ms, _g, {"z": _ZF, "x": _XZ, "p": _PZ}[_c]))  # P1 self-zeroing variants
    for _nm, _ly, _g, _vs in (("uk-s4", "unpooler", (1, 1, 8, 2, 4, 1, 0, 2), "zx"),    # n=1024: 64 x 4 = 256 CTAs x 256 thr
                              ("uk-s4w4", "unpooler", (1, 1, 4, 4, 4, 1, 0, 4), "x")):    # 256 CTAs x 128 thr
        for _c in _vs:
            GRD.append((_nm + _c, _ly, (1, 8), _g, 1 | {"z": _ZF, "x": _XZ}[_c]))
    GXP += GRD
    GXP += [("q144k16", "qkv_gate", (1, 8), (1, 1, 16, 8, 1, 1, 2), 1 | 2 << 12)]
    for _nm, _ly, _m, _g, _v in (("p", "pooler", 16, (1, 2, 8, 16, 1, 1, 0), 1), ("p", "pooler", 32, (1, 1, 8, 16, 1, 1, 2), 1),
                                 ("u", "unpooler", 16, (1, 1, 4, 16, 1, 1, 2), 1), ("u", "unpooler", 32, (1, 1, 4, 16, 1, 1, 2), 0),
                                 ("q", "qkv_gate", 16, (1, 1, 4, 8, 4, 1, 2, 1), 1), ("q", "qkv_gate", 32, (1, 1, 4, 16, 2, 1, 2, 1), 1)):
        for _sfx, _b in ():  # pw17: pw16's x-policy paths all lost (GCX); not built
            GXP.append(("gc-%s%s" % (_nm, _sfx), _ly, (_m,), _g, _v | _b))
    _OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
    _SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))
    def _c12(fr, dv=1):
        T, KC = fr.shape[:2]
        dev = fr.device
        b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
        h7, sg = (b >> 8) & 127, b >> 15
        win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
        B = int(win.argmax())
        d = h7 - B
        inw = (d >= 0) & (d < 8)
        d = torch.where(inw, d, 0).long()
        C = ((d << torch.tensor(_OS[dv], device=dev)) | (sg.long() << torch.tensor(_SS[dv], device=dev))).sum(-1)
        C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
        lo = (b & 255).to(torch.uint8)
        p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
        data = torch.stack((p0, p1, p2), 2).contiguous()
        ei = (~inw).nonzero()
        esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
        tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
        if not torch.equal(_unc12(data, tab, esc, dv), b):
            return None
        return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc
    def _unc12(data, tab, esc, dv):
        T, KC = data.shape[:2]
        dev = data.device
        w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
        L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
        C = w[:, :, 2].permute(0, 1, 3, 2)
        off = (C[..., None] >> torch.tensor(_OS[dv], device=dev)) & 7
        sgn = (C[..., None] >> torch.tensor(_SS[dv], device=dev)) & 1
        tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
        hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
        lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
        val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
        if len(esc):
            t, c, j, l, vv, x = esc.t()
            val[t, c, j, l, vv] ^= x << 8
        return val.int()
    PREPARE_BUDGET_S = 280.0
    MARGIN_RQ128 = 0.98  # pw52: a variant must beat the proven m=128 pick by 2% in-bake
    RQPICK128 = {("pooler", 128): "d2s5hzwP", ("qkv_gate", 128): "d2s5hzwP", ("unpooler", 128): "d2s5hzwP"}
    _EYE = {}
    FLUSH_MB = 128  # >= 2x L2 (50 MB on the H100/H200)
    FLUSH_BLOCKS, FLUSH_THREADS = 1056, 256
    HBM_RATE_TBPS = 3.35  # a cold probe must not beat this (judge_sim.HBM_RATE_TBPS)
    PROBE_BYTES = 6_300_032
    DISCRIM_BYTES = 16 << 20  # the probe the flush is validated on (6.3 MB is latency-bound and cannot discriminate)
    PAIRS = 8  # timed pairs per (candidate, baseline) comparison
    FLUSH_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// READ flush: every 16-byte line is loaded (default .ca policy, so it allocates in L2 and evicts whatever was there).
// The XOR accumulator is kept live by a comparison that is never true in practice, so no store is needed.
__global__ void flush_read_kernel(const uint4* __restrict__ p, long n4, float* sink) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  unsigned a = 0u, b = 0u, c = 0u, d = 0u;
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    a ^= v.x; b ^= v.y; c ^= v.z; d ^= v.w;
  }
  if ((a & b & c & d) == 0xFFFFFFFFu) sink[0] = 1.0f;
}

// probe: same access pattern, result written so the loads are live.
__global__ void probe_read_kernel(const uint4* __restrict__ p, long n4, float* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  unsigned a = 0u, b = 0u, c = 0u, d = 0u;
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    a ^= v.x; b ^= v.y; c ^= v.z; d ^= v.w;
  }
  if ((a & b & c & d) == 0xFFFFFFFFu) out[0] = 1.0f;
}

// empty kernel for the graph fixed-cost measurement (132 CTAs, like the judge's probe graph).
__global__ void nop_kernel(float* p) {
  if (threadIdx.x == 1024) p[0] = 1.0f;
}

void flush_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads) {
  long n4 = (long)(buf.numel() * buf.element_size() / 16);
  flush_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), n4, sink.data_ptr<float>());
}
void probe_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads) {
  long n4 = (long)(buf.numel() * buf.element_size() / 16);
  probe_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), n4, out.data_ptr<float>());
}
void nop(torch::Tensor out, int64_t blocks, int64_t threads) {
  nop_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>());
}
"""
    FLUSH_CPP = r"""
#include <torch/extension.h>
void flush_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads);
void probe_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads);
void nop(torch::Tensor out, int64_t blocks, int64_t threads);
"""
    _flext = [None]
    def _flushext():
        if _flext[0] is None:
            t0 = time.time()
            _flext[0] = load_inline(name="pw57_flush", cpp_sources=FLUSH_CPP, cuda_sources=FLUSH_CUDA,
                                    functions=["flush_read", "probe_read", "nop"],
                                    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
            print("BUILD pw57_flush %.1fs" % (time.time() - t0), flush=True)
        return _flext[0]
    class _Flusher:

        def __init__(self, mb=FLUSH_MB):
            self.ext = _flushext()
            self.buf = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
            self.sink = torch.zeros((), dtype=torch.float32, device="cuda")

        def __call__(self):
            self.ext.flush_read(self.buf, self.sink, FLUSH_BLOCKS, FLUSH_THREADS)
    _FL = [None]  # the flusher
    _OV = [0.0]   # graph fixed cost (us): the judge's 2*one_node - two_node
    _LAST = [None]  # (ratio, base_med, cand_med, ov) of the last paired comparison, for the BAKE lines
    def _flush():
        if _FL[0] is None:
            _FL[0] = _Flusher()
        return _FL[0]
    def _ev_median(fn, reps=20, fl=True):
        ts = []
        for _ in range(reps):
            if fl:
                _flush()()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]
    def _flush_validate():
        ext = _flushext()
        _flush()
        out = torch.zeros((), dtype=torch.float32, device="cuda")
        print("FLUSH_VALIDATE probes %.2f MB and %.2f MB; cold must be <= %.2f TB/s (pick on the larger probe)"
              % (PROBE_BYTES / 1e6, DISCRIM_BYTES / 1e6, HBM_RATE_TBPS), flush=True)
        ok_all = True
        for nb in (PROBE_BYTES, DISCRIM_BYTES):
            buf = torch.empty(nb, dtype=torch.uint8, device="cuda")
            buf.view(torch.float32).normal_()
            fn = lambda: ext.probe_read(buf, out, 528, 256)
            t = _ev_median(fn, fl=False)
            print("FLUSH_VALIDATE %-16s %6.2f MB %7.2f us %6.2f TB/s" % ("warm(no flush)", nb / 1e6, t,
                                                                         nb / 1e9 / (t * 1e-6) / 1e3), flush=True)
            t = _ev_median(fn)
            tb = nb / 1e9 / (t * 1e-6) / 1e3
            ok = tb <= HBM_RATE_TBPS
            if nb == DISCRIM_BYTES:
                ok_all = ok
            print("FLUSH_VALIDATE %-16s %6.2f MB %7.2f us %6.2f TB/s  %s"
                  % ("read/%dMB" % FLUSH_MB, nb / 1e6, t, tb, "COLD ok" if ok else "WARM reject"), flush=True)
        print("FLUSH_PICK read/%d %s" % (FLUSH_MB, "ok" if ok_all else "WARNING no cold flush"), flush=True)
        return ok_all
    def _capture(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            fn()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return g
    def _replay(graphs, pairs):
        names = list(graphs)
        ev = {n: [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(pairs)] for n in names}
        for p in range(pairs):
            for n in (names if p % 2 == 0 else names[::-1]):
                _flush()()
                a, b = ev[n][p]
                a.record(); graphs[n].replay(); b.record()
        torch.cuda.synchronize()
        return {n: [a.elapsed_time(b) * 1e3 for a, b in ev[n]] for n in names}
    def _fixed_cost(pairs=PAIRS):
        ext = _flushext()
        out = torch.zeros((), dtype=torch.float32, device="cuda")
        one = _capture(lambda: ext.nop(out, 132, 128))
        two = _capture(lambda: (ext.nop(out, 132, 128), ext.nop(out, 132, 128)))
        t = _replay({"one": one, "two": two}, pairs)
        o, w = statistics.median(t["one"]), statistics.median(t["two"])
        _OV[0] = 2 * o - w
        print("FIXED_COST one_node %.2f us two_node %.2f us overhead %.2f us" % (o, w, _OV[0]), flush=True)
        return _OV[0]
    def _paired(fn, base, pairs=PAIRS):
        ga, gb = _capture(fn), _capture(base)
        t = _replay({"a": ga, "b": gb}, pairs)
        ta, tb = t["a"], t["b"]
        ov = _OV[0]
        lim = 0.5 * min(statistics.median(ta), statistics.median(tb))
        if ov > lim:  # never let the subtraction reach the times (a bad fixed-cost read must not blow up a ratio)
            print("PAIR ov %.2f clamped to %.2f (times %.2f/%.2f)" % (ov, lim, statistics.median(ta),
                                                                     statistics.median(tb)), flush=True)
            ov = max(0.0, lim)
        r = [(b - ov) / max(a - ov, 1e-3) for a, b in zip(ta, tb)]  # (base-ov)/(cand-ov), the judge's r
        pr = [math.sqrt(r[i] * r[i + 1]) for i in range(0, len(r) - 1, 2)]
        return statistics.median(pr), statistics.median(tb), statistics.median(ta), ov
    def _time_us(fn, x, out, flush, sink, it=15, base=None):
        if base is None:
            g = _capture(lambda: fn(x, out))
            ts = _replay({"a": g}, it)["a"]
            ts.sort()
            return ts[len(ts) // 2]
        ratio, bmed, cmed, ov = _paired(lambda: fn(x, out), lambda: base(x, out), max(2, it // 2))
        _LAST[0] = (ratio, bmed, cmed, ov)
        return bmed / ratio
    RWIN128 = 1.01   # pw59 (wf_bake3): a proven pick is a group's reference only if the box does not read it slower than
    def _refsel(prov_name, prov_med, base_name, base_med, window):
        if prov_name is not None and prov_med is not None and prov_med <= base_med * window:
            return prov_name, prov_med, True
        return base_name, base_med, False
    def _pickrule(ref, cands, margin):
        best = None
        for nm, plan, r in cands:
            if best is None or r > best[2]:
                best = (nm, plan, r)
        if best is not None and best[2] > 1.0 / margin:
            return best[0], best[1], ref[2] / best[2]
        return ref
    RWV = {100: (256, 4, 4, 128, 128, 1), 101: (256, 8, 8, 128, 128, 1), 102: (128, 4, 4, 128, 128, 2), 103: (512, 8, 6, 128, 128, 1),
           110: (256, 4, 4, 256, 256, 1), 111: (256, 8, 8, 256, 256, 1), 112: (128, 4, 4, 256, 256, 2), 113: (256, 8, 8, 256, 128, 1),
           114: (512, 8, 4, 256, 256, 1)}
    RWV = {k: v + (0, 1) for k, v in RWV.items()}  # + (MODE, CL)
    RWV.update({104: (256, 4, 3, 128, 128, 2, 0, 1), 105: (256, 4, 2, 128, 128, 2, 0, 1), 106: (256, 4, 2, 128, 128, 2, 4, 1),
                120: RWV[100][:6] + (4, 1), 121: RWV[100][:6] + (0, 2), 123: RWV[100][:6] + (4, 2), 126: RWV[101][:6] + (4, 1),
                127: RWV[102][:6] + (4, 1), 107: (256, 4, 3, 128, 128, 2, 4, 1), 140: (256, 4, 3, 128, 128, 2, 4, 2),
                141: (256, 4, 3, 128, 128, 2, 0, 2), 142: RWV[101][:6] + (4, 2),
                150: RWV[101][:6] + (12, 1), 151: (256, 4, 3, 128, 128, 2, 12, 1), 152: (256, 4, 4, 128, 128, 1, 12, 1),
                130: RWV[113][:6] + (4, 1), 131: RWV[113][:6] + (0, 2), 136: RWV[112][:6] + (4, 1)})
    RWV.update({108: (512, 8, 3, 128, 128, 2, 0, 1)})
    RWST = (101, 104, 107, 103, 108, 113, 130)  # MODE 16 (globaltimer stamps) instances: variant + 20000
    RWV.update({v + 20000: RWV[v][:6] + (RWV[v][6] | 16, RWV[v][7]) for v in RWST})
    RTV = {200: (512, 6, 128, 128, 1, 4), 201: (256, 3, 128, 128, 2, 4), 202: (256, 4, 256, 128, 1, 4), 203: (512, 4, 256, 128, 1, 4),
           204: (1024, 6, 128, 128, 1, 4), 206: (512, 6, 128, 128, 1, 0), 210: (256, 4, 128, 128, 1, 4), 211: (256, 3, 128, 128, 2, 4),
           230: (512, 6, 128, 128, 1, 32), 231: (256, 3, 128, 128, 2, 32), 232: (256, 4, 128, 128, 1, 32),
           240: (512, 6, 128, 128, 1, 68), 241: (256, 3, 128, 128, 2, 68), 242: (256, 4, 128, 128, 1, 68),
           250: (128, 3, 128, 128, 2, 4), 251: (256, 3, 128, 128, 2, 132), 252: (512, 3, 128, 128, 2, 4),
           253: (256, 3, 128, 128, 2, 196), 254: (128, 3, 128, 128, 2, 132), 255: (512, 3, 128, 128, 2, 132),
           256: (256, 3, 128, 128, 1, 548), 257: (512, 3, 128, 128, 1, 548), 258: (512, 3, 128, 128, 1, 32),
           261: (256, 3, 128, 128, 2, 36), 262: (256, 3, 128, 128, 2, 548),
           263: (256, 4, 128, 128, 2, 36), 264: (512, 3, 128, 128, 2, 36),
           265: (256, 3, 128, 128, 2, 1028), 266: (256, 3, 128, 128, 2, 1156), 267: (256, 3, 128, 128, 2, 1220),
           268: (256, 4, 128, 128, 1, 132), 269: (256, 4, 128, 128, 1, 1156),
           270: (256, 3, 128, 128, 2, 2180), 271: (256, 3, 128, 128, 2, 6276), 272: (256, 3, 128, 128, 2, 2052),
           273: (256, 3, 128, 128, 2, 6148), 274: (512, 3, 128, 128, 2, 6276),
           275: (256, 3, 128, 128, 2, 3076), 276: (256, 3, 128, 128, 2, 2116), 277: (256, 4, 128, 128, 1, 2052),
           278: (256, 3, 64, 64, 1, 4),  # TOKS 64 / WN 64: MINB 2 needs >168 regs (ptxas C7600), MINB 1 = 2 waves at m=128
           282: (512, 6, 128, 128, 1, 2052), 283: (512, 6, 128, 128, 1, 3076), 284: (512, 6, 128, 128, 1, 6148),
           285: (256, 2, 128, 128, 2, 132), 286: (256, 2, 128, 128, 2, 4),
           288: (256, 3, 128, 128, 2, 3204), 289: (256, 3, 128, 128, 2, 3140), 290: (256, 4, 128, 128, 1, 1156)}
    RTV.update({v + 20000: c[:5] + (c[5] | 16,) for v, c in list(RTV.items())})
    RWLAD = True  # pw29h ladder instances: variant + 1000 * MODE (1 = loads only, 2 = no reds, 3 = loads only, no reds)
    if RWLAD:
        RWV.update({k + 1000 * md: RWV[k][:6] + (md, 1) for k in (101, 104) for md in (1, 2, 3)})
    RQV = [  # (name, NT, ST, GS, PRB, WM, INS, FL): ST k64 slots (W 12 KB + x NT*128 B each), GS k64 stages per consumer
        ("f12", 128, 8, 1, 5, 0, 0, 0),
        ("f12x", 128, 8, 1, 6, 0, 0, 0),
        ("s0", 128, 8, 1, 1, 0, 0, 0),
        ("s1", 128, 8, 1, 2, 0, 0, 0),
        ("s2", 128, 8, 1, 3, 0, 0, 0),
        ("s3", 128, 8, 1, 4, 0, 0, 0),
        ("s3g2", 128, 8, 2, 4, 0, 0, 0),
        ("s3m1", 128, 8, 1, 4, 1, 0, 0),
        ("s3m2", 128, 8, 1, 4, 2, 0, 0),
        ("s3m2s7", 128, 7, 1, 4, 2, 0, 0),
        ("s3s6", 128, 6, 1, 4, 0, 0, 0),
        ("s3s5", 128, 5, 1, 4, 0, 0, 0),
        ("s3s4", 128, 4, 1, 4, 0, 0, 0),
        ("s3s6h", 128, 6, 1, 4, 0, 0, 1),
        ("s3s5h", 128, 5, 1, 4, 0, 0, 1),
        ("s3s6g2", 128, 6, 2, 4, 0, 0, 0),
        ("s1s6", 128, 6, 1, 2, 0, 0, 0),
        ("s1s5", 128, 5, 1, 2, 0, 0, 0),
        ("s4s6z", 128, 6, 1, 0, 0, 0, 2),
        ("s4s5z", 128, 5, 1, 0, 0, 0, 2),
        ("s4s6hz", 128, 6, 1, 0, 0, 0, 3),
        ("s4s5hz", 128, 5, 1, 0, 0, 0, 3),
        ("s4s5hzi", 128, 5, 1, 0, 0, 1, 3),
        ("s4s5zi", 128, 5, 1, 0, 0, 1, 2),
        ("s4s5zni", 128, 5, 1, 0, 0, 1, 2097154),
        ("s4s5zwi", 128, 5, 1, 0, 0, 1, 9437186),
        ("s4hz", 128, 8, 1, 0, 0, 0, 3),
        ("n256s5hz", 256, 5, 1, 0, 0, 0, 3),
        ("n256s4hz", 256, 4, 1, 0, 0, 0, 3),
        ("n256s5z", 256, 5, 1, 0, 0, 0, 2),
        ("n256s5m1hzmEH", 256, 5, 1, 0, 1, 0, 65603),
        ("n256s5m1hzm", 256, 5, 1, 0, 1, 0, 67),
        ("s3s7", 128, 7, 1, 4, 0, 0, 0),
        ("s3h", 128, 8, 1, 4, 0, 0, 1),
        ("s3m2h", 128, 8, 1, 4, 2, 0, 1),
        ("s4m2z", 128, 8, 1, 0, 2, 0, 3),
        ("r8m2", 128, 8, 1, 8, 2, 0, 0),
        ("s3m2i", 128, 8, 1, 4, 2, 1, 0),
        ("s3g2m1", 128, 8, 2, 4, 1, 0, 0),
        ("s4", 128, 8, 1, 0, 0, 0, 1),
        ("s4m1", 128, 8, 1, 0, 1, 0, 1),
        ("s4z", 128, 8, 1, 0, 0, 0, 3),
        ("s4m1z", 128, 8, 1, 0, 1, 0, 3),
        ("s4g2m1z", 128, 8, 2, 0, 1, 0, 3),
        ("ep", 128, 8, 1, 7, 0, 0, 0),
        ("epst", 128, 8, 1, 7, 0, 0, 4),
        ("ep0", 128, 8, 1, 7, 0, 0, 8),
        ("s4st", 128, 8, 1, 0, 0, 0, 5),
        ("s4zi", 128, 8, 1, 0, 0, 1, 3),
        ("r8", 128, 8, 1, 8, 0, 0, 0),
        ("r10", 128, 8, 1, 10, 0, 0, 0),  # wf_dilp: constant-A tensor floor probe
        ("r8m1", 128, 8, 1, 8, 1, 0, 0),
        ("s3i", 128, 8, 1, 4, 0, 1, 0),
        ("s3m1i", 128, 8, 1, 4, 1, 1, 0),
        ("s4m1zi", 128, 8, 1, 0, 1, 1, 3),
        ("t32s5z", 32, 5, 1, 0, 0, 0, 2),
        ("t32s5hz", 32, 5, 1, 0, 0, 0, 3),
        ("t32s8z", 32, 8, 1, 0, 0, 0, 2),
        ("t32s8hz", 32, 8, 1, 0, 0, 0, 3),
        ("t32s12z", 32, 12, 1, 0, 0, 0, 2),
        ("t32s12hz", 32, 12, 1, 0, 0, 0, 3),
        ("t32g2s8z", 32, 8, 2, 0, 0, 0, 2),
        ("t32m2s8z", 32, 8, 1, 0, 2, 0, 2),
        ("t32s8zi", 32, 8, 1, 0, 0, 1, 2),
        ("h32s8", 32, 8, 1, 0, 0, 0, 20),
        ("h32s8z", 32, 8, 1, 0, 0, 0, 18),
        ("h32s12", 32, 12, 1, 0, 0, 0, 20),
        ("h32s12z", 32, 12, 1, 0, 0, 0, 18),
        ("t16s5z", 16, 5, 1, 0, 0, 0, 2),
        ("t16s5hz", 16, 5, 1, 0, 0, 0, 3),
        ("t16s8z", 16, 8, 1, 0, 0, 0, 2),
        ("t16s8hz", 16, 8, 1, 0, 0, 0, 3),
        ("t16s12z", 16, 12, 1, 0, 0, 0, 2),
        ("t16s12hz", 16, 12, 1, 0, 0, 0, 3),
        ("t16g2s8z", 16, 8, 2, 0, 0, 0, 2),
        ("t16m2s8z", 16, 8, 1, 0, 2, 0, 2),
        ("t16s8zi", 16, 8, 1, 0, 0, 1, 2),
        ("h16s8", 16, 8, 1, 0, 0, 0, 20),
        ("h16s8z", 16, 8, 1, 0, 0, 0, 18),
        ("h16s12", 16, 12, 1, 0, 0, 0, 20),
        ("h16s12z", 16, 12, 1, 0, 0, 0, 18),
        ("t32s3z", 32, 3, 1, 0, 0, 0, 2),
        ("t32s3hz", 32, 3, 1, 0, 0, 0, 3),
        ("t32s4z", 32, 4, 1, 0, 0, 0, 2),
        ("t32s4hz", 32, 4, 1, 0, 0, 0, 3),
        ("t32s5hzi", 32, 5, 1, 0, 0, 1, 3),
        ("t16s3z", 16, 3, 1, 0, 0, 0, 2),
        ("t16s3hz", 16, 3, 1, 0, 0, 0, 3),
        ("t16s4z", 16, 4, 1, 0, 0, 0, 2),
        ("t16s4hz", 16, 4, 1, 0, 0, 0, 3),
        ("t16s5hzi", 16, 5, 1, 0, 0, 1, 3),
        ("t32s5hze", 32, 5, 1, 0, 0, 0, 35),
        ("t32s5ze", 32, 5, 1, 0, 0, 0, 34),
        ("t32s6hze", 32, 6, 1, 0, 0, 0, 35),
        ("t32s6hz", 32, 6, 1, 0, 0, 0, 3),
        ("t32s5hzei", 32, 5, 1, 0, 0, 1, 35),
        ("t32s6hzei", 32, 6, 1, 0, 0, 1, 35),
        ("t16s5hze", 16, 5, 1, 0, 0, 0, 35),
        ("t16s5ze", 16, 5, 1, 0, 0, 0, 34),
        ("t16s6hze", 16, 6, 1, 0, 0, 0, 35),
        ("t16s6hz", 16, 6, 1, 0, 0, 0, 3),
        ("t16s5hzei", 16, 5, 1, 0, 0, 1, 35),
        ("t16s6hzei", 16, 6, 1, 0, 0, 1, 35),
        ("h128s8e", 128, 8, 1, 0, 0, 0, 52),
        ("h128s5e", 128, 5, 1, 0, 0, 0, 52),  # wf_rq128b: HALF at ST 5 (72.7 KB smem: 2 CTAs/SM, launches)
        ("h32s4e", 32, 4, 1, 0, 0, 0, 52),
        ("h32s5e", 32, 5, 1, 0, 0, 0, 52),
        ("s4s5hze", 128, 5, 1, 0, 0, 0, 35),
        ("s4s5hzem", 128, 5, 1, 0, 0, 0, 99),
        ("s4s5hzer", 128, 5, 1, 0, 0, 0, 163),
        ("s4s5hzemr", 128, 5, 1, 0, 0, 0, 227),
        ("s4s5hzen", 128, 5, 1, 0, 0, 0, 547),
        ("s4s5hzem2", 128, 5, 1, 0, 2, 0, 35),
        ("s4s5hzemf", 128, 5, 1, 0, 0, 0, 16483),
        ("s4s5hzemi", 128, 5, 1, 0, 0, 1, 99),
        ("s4s5hzeri", 128, 5, 1, 0, 0, 1, 163),
        ("s4s5hzemfi", 128, 5, 1, 0, 0, 1, 16483),
        ("s4s5hzew", 128, 5, 1, 0, 0, 0, 32803),
        ("s4s5hzewi", 128, 5, 1, 0, 0, 1, 32803),
        ("s4s5hzemw", 128, 5, 1, 0, 0, 0, 32867),
        ("s4s5hzemwi", 128, 5, 1, 0, 0, 1, 32867),
        ("s4s5hzeni", 128, 5, 1, 0, 0, 1, 547),
        ("h128s5", 128, 5, 1, 0, 0, 0, 20),
        ("h128s5z", 128, 5, 1, 0, 0, 0, 18),
        ("h128s8", 128, 8, 1, 0, 0, 0, 20),
        ("h128s8z", 128, 8, 1, 0, 0, 0, 18),
        ("s4s5zw", 128, 5, 1, 0, 0, 0, 9437186),
        ("s4s5hzw", 128, 5, 1, 0, 0, 0, 9437187),
        ("s4s5zn", 128, 5, 1, 0, 0, 0, 2097154),
        ("s4s5hzn", 128, 5, 1, 0, 0, 0, 2097155),
        ("s4s5zEH", 128, 5, 1, 0, 0, 0, 65538),
        ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),
        ("t32s5hzw", 32, 5, 1, 0, 0, 0, 9437187),
        ("t32s6hzw", 32, 6, 1, 0, 0, 0, 9437187),
        ("t32s5zw", 32, 5, 1, 0, 0, 0, 9437186),
        ("h32s5hzw", 32, 5, 1, 0, 0, 0, 9437204),
        ("h32s4hzw", 32, 4, 1, 0, 0, 0, 9437204),
        ("t32s5hzwE", 32, 5, 1, 0, 0, 0, 9437219),
        ("t32s6hzwE", 32, 6, 1, 0, 0, 0, 9437219),
        ("t32s5hzn", 32, 5, 1, 0, 0, 0, 2097155),
        ("t32s6hzn", 32, 6, 1, 0, 0, 0, 2097155),
        ("t32s5zn", 32, 5, 1, 0, 0, 0, 2097154),
        ("h32s5hzn", 32, 5, 1, 0, 0, 0, 2097156),
        ("h32s4hzn", 32, 4, 1, 0, 0, 0, 2097156),
        ("t32s5hzwi", 32, 5, 1, 0, 0, 1, 9437187),
        ("t32s5hzni", 32, 5, 1, 0, 0, 1, 2097155),
        ("t16s4hzw", 16, 4, 1, 0, 0, 0, 9437187),
        ("t16s5hzw", 16, 5, 1, 0, 0, 0, 9437187),
        ("h16s5hzw2", 16, 5, 1, 0, 0, 0, 9438227),
        ("t16s4hzn", 16, 4, 1, 0, 0, 0, 2097155),
        ("t16s5hzn", 16, 5, 1, 0, 0, 0, 2097155),
        ("t16s5hzwi", 16, 5, 1, 0, 0, 1, 9437187),
        ("t16s5hzni", 16, 5, 1, 0, 0, 1, 2097155),
        ("s4s6hze", 128, 6, 1, 0, 0, 0, 35),
        ("s4s7hze", 128, 7, 1, 0, 0, 0, 35),
        ("s4s8hze", 128, 8, 1, 0, 0, 0, 35),
        ("s4s8hzeEH", 128, 8, 1, 0, 0, 0, 65571),
        ("s4s5hzeEH", 128, 5, 1, 0, 0, 0, 65571),
        ("s4s6hzeEH", 128, 6, 1, 0, 0, 0, 65571),
        ("s4s5hzem1", 128, 5, 1, 0, 1, 0, 35),
        ("s4s5hzem2", 128, 5, 1, 0, 2, 0, 35),
        ("h128s6e", 128, 6, 1, 0, 0, 0, 52),
        ("h128s10e", 128, 10, 1, 0, 0, 0, 52),
        ("h128s8eg2", 128, 8, 2, 0, 0, 0, 52),
        ("h128s10eg2", 128, 10, 2, 0, 0, 0, 52),
        ("h128s8em1", 128, 8, 1, 0, 1, 0, 52),
        ("s4s5hze2", 128, 5, 2, 0, 0, 0, 35),
        ("s4s6hze2", 128, 6, 2, 0, 0, 0, 35),
        ("s4s5hzei", 128, 5, 1, 0, 0, 1, 35),
        ("h128s8ei", 128, 8, 1, 0, 0, 1, 52),
        ("h32s5hze2", 32, 5, 1, 0, 0, 0, 1075),
        ("h32s5hze2i", 32, 5, 1, 0, 0, 1, 1075),
        ("h32s6hze2", 32, 6, 1, 0, 0, 0, 1075),
        ("h32s8hze2", 32, 8, 1, 0, 0, 0, 1075),
        ("h32s5ze2", 32, 5, 1, 0, 0, 0, 1074),
        ("h32s4hze2", 32, 4, 1, 0, 0, 0, 1075),
        ("h32s7hze2", 32, 7, 1, 0, 0, 0, 1075),
        ("f32s5hze2", 32, 5, 1, 0, 0, 0, 2083),
        ("f32s5hze2i", 32, 5, 1, 0, 0, 1, 2083),
        ("h16s5hze2", 16, 5, 1, 0, 0, 0, 1075),
        ("h16s5hze2i", 16, 5, 1, 0, 0, 1, 1075),
        ("h16s8hze2", 16, 8, 1, 0, 0, 0, 1075),
        ("h16s4hze2", 16, 4, 1, 0, 0, 0, 1075),
        ("h16s6hze2", 16, 6, 1, 0, 0, 0, 1075),
        ("h16s7hze2", 16, 7, 1, 0, 0, 0, 1075),
        ("f16s5hze2", 16, 5, 1, 0, 0, 0, 2083),
        ("d1s5hze", 128, 5, 1, 0, 0, 0, 4131),
        ("d2s5hze", 128, 5, 1, 0, 0, 0, 12323),
        ("d2s5hzw", 128, 5, 1, 0, 0, 0, 9449507),  # pw60: d2s5hze + FL bit 20|23 (cursor)
        ("d2s5hzwP", 128, 5, 1, 0, 0, 0, 42999939),  # d2s5hzw + FL bit 25 (9449507 | 33554432)
        ("s4s5hzwP", 128, 5, 1, 0, 0, 0, 42991619),  # s4s5hzw + FL bit 25 (9437187 | 33554432)
        ("d1s5hzei", 128, 5, 1, 0, 0, 1, 4131),
        ("d2s5hzei", 128, 5, 1, 0, 0, 1, 12323),
        ("d0s5hzen", 128, 5, 1, 0, 0, 0, 4643),
        ("d0s5hzeni", 128, 5, 1, 0, 0, 1, 4643),
        ("t32d1s5hze", 32, 5, 1, 0, 0, 0, 4131),
        ("t32d2s5hze", 32, 5, 1, 0, 0, 0, 12323),
        ("t32d1s5hzei", 32, 5, 1, 0, 0, 1, 4131),
        ("m2s3m1hz", 128, 3, 2, 0, 1, 0, 3),
        ("m2s4m1hz", 128, 4, 2, 0, 1, 0, 3),
        ("m2s3m1h", 128, 3, 2, 0, 1, 0, 1),
        ("m2s4m1h", 128, 4, 2, 0, 1, 0, 1),
        ("m2s3m1hze", 128, 3, 2, 0, 1, 0, 35),
        ("m2s4m1hze", 128, 4, 2, 0, 1, 0, 35),
        ("m2s3m2hz", 128, 3, 2, 0, 2, 0, 3),
        ("m2s4m2hz", 128, 4, 2, 0, 2, 0, 3),
        ("m2s3m1z", 128, 3, 2, 0, 1, 0, 2),
        ("m2s4m1z", 128, 4, 2, 0, 1, 0, 2),
        ("m2s3m1zEH", 128, 3, 2, 0, 1, 0, 65538),
        ("m2s4m1zEH", 128, 4, 2, 0, 1, 0, 65538),
        ("m2s3m1hzEH", 128, 3, 2, 0, 1, 0, 65539),
        ("m2s4m1hzEH", 128, 4, 2, 0, 1, 0, 65539),
        ("g2s5z", 128, 5, 2, 0, 0, 0, 2),
        ("g2s5hz", 128, 5, 2, 0, 1, 0, 3),
        ("g2s5hze", 128, 5, 2, 0, 1, 0, 35),
        ("g2s6hz", 128, 6, 2, 0, 1, 0, 3),
        ("g2s4hz", 128, 4, 2, 0, 1, 0, 3),
        ("s4s6hzew1", 128, 6, 1, 0, 1, 0, 35),
        ("s4s7hzew1", 128, 7, 1, 0, 1, 0, 35),
        ("s4s8hzew1", 128, 8, 1, 0, 1, 0, 35),
        ("m2s3m1hzi", 128, 3, 2, 0, 1, 1, 3),
        ("m2s4m1hzi", 128, 4, 2, 0, 1, 1, 3),
        ("d2s5z", 128, 5, 1, 0, 0, 0, 12322),
        ("d2s4hze", 128, 4, 1, 0, 0, 0, 12323),
        ("d2s6hze", 128, 6, 1, 0, 0, 0, 12323),
        ("d2s5hzem2", 128, 5, 1, 0, 2, 0, 12323),
        ("d2s3hze", 128, 3, 1, 0, 0, 0, 12323),
        ("d2s4z", 128, 4, 1, 0, 0, 0, 12322),
        ("t32s4hze", 32, 4, 1, 0, 0, 0, 35),
        ("t32s7hze", 32, 7, 1, 0, 0, 0, 35),
        ("t16s4hze", 16, 4, 1, 0, 0, 0, 35),
        ("t16s7hze", 16, 7, 1, 0, 0, 0, 35),
    ]
    RQMS = {nm: 2 for nm in ("m2s3m1hz", "m2s4m1hz", "m2s3m1h", "m2s4m1h", "m2s3m1hze", "m2s4m1hze", "m2s3m2hz", "m2s4m2hz",
                            "m2s3m1z", "m2s4m1z", "m2s3m1zEH", "m2s4m1zEH", "m2s3m1hzEH", "m2s4m1hzEH",
                            "m2s3m1hzi", "m2s4m1hzi")}
    RQI = {v[0]: i for i, v in enumerate(RQV)}
    RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # judge 1e-2 (rows 5x)
    RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
    RQPROD = ["s4s5z", "s4s5hz", "s4s5hze", "d2s5hze", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]
    RQI = {v[0]: i for i, v in enumerate(RQV)}
    RQCUR = {RQI[n] for n in ("s4s5zw", "s4s5hzw", "d2s5hzwP", "s4s5hzwP", "t32s5hzw", "t32s6hzw", "t32s5zw", "h32s5hzw", "h32s4hzw",
                              "t32s5hzwE", "t32s6hzwE", "t32s5hzwi",
                              "t16s4hzw", "t16s5hzw", "h16s5hzw2", "t16s5hzwi")}  # variants that need the FL bit 20 cursor array
    RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # judge 1e-2 (rows 5x)
    RQCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  const uint32_t* esc2;              // FL bit 20: per-(warp,lane) sub-lists [KS][NB][8][32][4], sentinel-terminated
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
// wf_rq128b: relaxed (non-release) arrive. The x/W slots are only ever *read* by the consumer (LDS + wgmma, the latter
// retired by wait_group), so the release ordering of the default arrive -- which makes the arrive wait for the LDS just
// issued (P4: 95 clk) -- is not needed.
__device__ __forceinline__ void rq_arrive_r(uint32_t b) { asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// wf_dilp DILP (FL bit 12): the same 36 ALU ops, written as four independent 3-deep chains: every (hl, hh) pair is
// computed first (8 words, 20 ops, all independent), then the 16 PRMTs (all independent). rq_unit's per-dec form makes the
// compiler walk the four sub-blocks in order (each PRMT waits on its own LDS->LOP3->IADD chain); here the four chains'
// latencies overlap. Same values, same order of A[] writes.
__device__ __forceinline__ void rq_unit_ilp(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  const uint32_t hl0 = (q2.x & 0x87878787u) + Tb, hh0 = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  const uint32_t hl1 = (q2.y & 0x87878787u) + Tb, hh1 = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  const uint32_t hl2 = (q2.z & 0x87878787u) + Tb, hh2 = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  const uint32_t hl3 = (q2.w & 0x87878787u) + Tb, hh3 = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
  A[0] = rq_prmt(q0.x, hl0, 0x5140); A[1] = rq_prmt(q0.x, hl0, 0x7362);
  A[2] = rq_prmt(q0.y, hh0, 0x5140); A[3] = rq_prmt(q0.y, hh0, 0x7362);
  A[4] = rq_prmt(q0.z, hl1, 0x5140); A[5] = rq_prmt(q0.z, hl1, 0x7362);
  A[6] = rq_prmt(q0.w, hh1, 0x5140); A[7] = rq_prmt(q0.w, hh1, 0x7362);
  A[8] = rq_prmt(q1.x, hl2, 0x5140); A[9] = rq_prmt(q1.x, hl2, 0x7362);
  A[10] = rq_prmt(q1.y, hh2, 0x5140); A[11] = rq_prmt(q1.y, hh2, 0x7362);
  A[12] = rq_prmt(q1.z, hl3, 0x5140); A[13] = rq_prmt(q1.z, hl3, 0x7362);
  A[14] = rq_prmt(q1.w, hh3, 0x5140); A[15] = rq_prmt(q1.w, hh3, 0x7362);
}
// wf_rq128b WK: apply one patch-table word (bit 15 valid, 11:8 reg, 7 half, 6:0 xor7) to a decoded A fragment. The
// predicate is per-lane but the *instruction count* is warp-uniform, so the wgmma.fence no longer waits for a walker.
__device__ __forceinline__ void rq_patch(uint32_t (&A)[16], uint32_t w) {
  if (w & 0x8000u) {
    const int r = (int)((w >> 8) & 15);
    const uint32_t xr = (w & 0x7fu) << (8 + ((w >> 7) & 1) * 16);
#pragma unroll
    for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
  }
}
// wf_rq128b FX2: the same walk with a fixed iteration count (warp-uniform instruction count, so the wgmma.fence of the
// warpgroup no longer waits for the warp that happens to walk the longest list). K must be >= the max hits per stage.
__device__ __forceinline__ void rq_fix2(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
#pragma unroll
  for (int it = 0; it < 2; it++) {
    if (nxs == (uint32_t)seq) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
      const bool me = lane == (int)((nx >> 12) & 31);
#pragma unroll
      for (int i = 0; i < 16; i++) if (me && i == r) A[i] ^= xr;
      ++ei;
      if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
      nx = __shfl_sync(0xffffffffu, ec, ei & 31);
      nxs = nx >> 17;
    }
  }
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
// wf_dilp WPF (FL bit 13): the same walk with the next entry's shfl issued before the current entry's 16 predicated XORs,
// so the shfl latency (the only dependent chain in the walk) hides under the apply instead of stalling the wgmma.fence.
__device__ __forceinline__ void rq_fixp(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    uint32_t nxn = __shfl_sync(0xffffffffu, ec, (ei + 1) & 31);
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) { ec = __ldg(el + ei + lane); nxn = __shfl_sync(0xffffffffu, ec, 0); }
    nx = nxn;
    nxs = nx >> 17;
  }
}
// FL bit 20: per-thread staged cursor.  E holds this thread's own escape entries sorted by stage (head = E.x),
// hs = E.x >> 17 kept pre-shifted so the common path is one compare + one branch; no shfl, no cross-lane dependency.
// The apply needs a runtime register index, so it is either 16 predicated XORs (if-converted -> 32 instrs EVERY stage,
// measured: loop body 2120 vs 1120 no-escape) or a real branch around them.  asm volatile cannot be if-converted, so
// rq_xor16 keeps the branch: common path = ISETP + BRA.  FL bit 22 = the __any_sync (warp-uniform branch) flavour.
__device__ __forceinline__ void rq_xor16(uint32_t (&A)[16], uint32_t nx) {
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16), r = (nx >> 8) & 15;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %16, 0;\n\t@p xor.b32 %0, %0, %17;\n\t"
      "setp.eq.u32 p, %16, 1;\n\t@p xor.b32 %1, %1, %17;\n\t"
      "setp.eq.u32 p, %16, 2;\n\t@p xor.b32 %2, %2, %17;\n\t"
      "setp.eq.u32 p, %16, 3;\n\t@p xor.b32 %3, %3, %17;\n\t"
      "setp.eq.u32 p, %16, 4;\n\t@p xor.b32 %4, %4, %17;\n\t"
      "setp.eq.u32 p, %16, 5;\n\t@p xor.b32 %5, %5, %17;\n\t"
      "setp.eq.u32 p, %16, 6;\n\t@p xor.b32 %6, %6, %17;\n\t"
      "setp.eq.u32 p, %16, 7;\n\t@p xor.b32 %7, %7, %17;\n\t"
      "setp.eq.u32 p, %16, 8;\n\t@p xor.b32 %8, %8, %17;\n\t"
      "setp.eq.u32 p, %16, 9;\n\t@p xor.b32 %9, %9, %17;\n\t"
      "setp.eq.u32 p, %16, 10;\n\t@p xor.b32 %10, %10, %17;\n\t"
      "setp.eq.u32 p, %16, 11;\n\t@p xor.b32 %11, %11, %17;\n\t"
      "setp.eq.u32 p, %16, 12;\n\t@p xor.b32 %12, %12, %17;\n\t"
      "setp.eq.u32 p, %16, 13;\n\t@p xor.b32 %13, %13, %17;\n\t"
      "setp.eq.u32 p, %16, 14;\n\t@p xor.b32 %14, %14, %17;\n\t"
      "setp.eq.u32 p, %16, 15;\n\t@p xor.b32 %15, %15, %17;\n\t}"
      : "+r"(A[0]), "+r"(A[1]), "+r"(A[2]), "+r"(A[3]), "+r"(A[4]), "+r"(A[5]), "+r"(A[6]), "+r"(A[7]),
        "+r"(A[8]), "+r"(A[9]), "+r"(A[10]), "+r"(A[11]), "+r"(A[12]), "+r"(A[13]), "+r"(A[14]), "+r"(A[15])
      : "r"(r), "r"(xr));
}
__device__ __forceinline__ void rq_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void rq_xor16s(uint32_t (&A)[16], uint32_t nx) {  // if-convertible 16-way select
  const int r = (int)((nx >> 8) & 15);
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
  for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
}
__device__ __forceinline__ void rq_fixc(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 20: real branch
  while (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
}
__device__ __forceinline__ void rq_fixv(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 22: warp-uniform
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16s(A, E.x); rq_adv(E, hs); }
  }
}
// bit 23: warp-uniform branch (__any_sync) around the asm apply -> the apply is neither if-converted nor hoisted out of
// the loop (micro: loop body 107 vs 101 no-escape, 115 vote+select, 199 asm alone), so wgmma.fence has nothing to wait for
__device__ __forceinline__ void rq_fixw(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
  }
}
// FL bit 25 (wf_escur3 H2b, ported): patch the 8 hi-byte registers instead of the 16 A registers.
// rq_unit_ilp builds A[4g+0]=PRMT(q0,hl_g,0x5140) A[4g+1]=PRMT(q0,hl_g,0x7362) A[4g+2]=PRMT(q0,hh_g,0x5140)
// A[4g+3]=PRMT(q0,hh_g,0x7362), so entry (reg, half) -> H[reg>>1], byte (reg&1)*2 + half => 8 predicated XORs
// instead of 16 (half the apply block, so the taken branch over it is cheap).
__device__ __forceinline__ void rq_hi8(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void rq_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = rq_prmt(q0.x, H[0], 0x5140); A[1] = rq_prmt(q0.x, H[0], 0x7362);
  A[2] = rq_prmt(q0.y, H[1], 0x5140); A[3] = rq_prmt(q0.y, H[1], 0x7362);
  A[4] = rq_prmt(q0.z, H[2], 0x5140); A[5] = rq_prmt(q0.z, H[2], 0x7362);
  A[6] = rq_prmt(q0.w, H[3], 0x5140); A[7] = rq_prmt(q0.w, H[3], 0x7362);
  A[8] = rq_prmt(q1.x, H[4], 0x5140); A[9] = rq_prmt(q1.x, H[4], 0x7362);
  A[10] = rq_prmt(q1.y, H[5], 0x5140); A[11] = rq_prmt(q1.y, H[5], 0x7362);
  A[12] = rq_prmt(q1.z, H[6], 0x5140); A[13] = rq_prmt(q1.z, H[6], 0x7362);
  A[14] = rq_prmt(q1.w, H[7], 0x5140); A[15] = rq_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void rq_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@

template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB;  // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);  // WK patch table
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
__global__ void __launch_bounds__((FL & 1024) ? 256 : 384, (FL & 3072) ? 2 : 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
  constexpr bool MRG = (FL & 64) != 0;    // one empty barrier per slot: the x release is folded into the W release
  constexpr bool RLAX = (FL & 128) != 0;  // relaxed (non-release) empty arrive
  constexpr bool NOE = (FL & 512) != 0;   // PROBE: no escape walk (wrong output, times the floor)
  constexpr bool FX2 = (FL & 16384) != 0;  // fixed-iteration uniform escape loop (bit 14: bit 10 is pw47's H2)
  constexpr bool WK = (FL & 32768) != 0;   // patch-table walker (warp PW+3) instead of the per-warp escape list walk (bit 15: bit 11 is pw47's F2)
  constexpr bool DILP = (FL & 4096) != 0; // wf_dilp: 4 independent decode chains (rq_unit_ilp)
  constexpr bool WPF = (FL & 8192) != 0;  // wf_dilp: escape walk with the next-entry shfl prefetched
  static_assert(!WK || (PRB == 0 && MS == 1), "WK: full kernel, MS == 1");  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && GS == MS && (ST >= 5 && ST <= 8) && PRB == 0 && WM <= 1 && INS == 0 && (FL == 65538 || FL == 65539 || FL == 65570 || FL == 65571))
                || (NT == 256 && GS == MS && ST == 5 && PRB == 0 && WM == 1 && INS == 0 && FL == 65603), "EH backport approved M128/ST5-8/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // wf_rq32b: 2 CTAs per SM. FL bit 10 (H2): 256-thread CTA = 4 math warps + W lane + x lane + 2 zeroing warps, HALF
  // (64 W rows) only, launch cap 128 regs, math 216 / producer 40. FL bit 11 (F2): 384 threads, full 128 rows, launch
  // cap 84 regs, math 104 / producer 40 (KS must double so the grid still fills 2 x 132 SMs).
  constexpr bool H2 = (FL & 1024) != 0, F2 = (FL & 2048) != 0;
  static_assert(!H2 || HALF, "H2 needs HALF (64 W rows, one math warpgroup)");
  constexpr int PW = H2 ? 4 : 8;                       // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)
  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  // MS (k64 stages per ring slot): one full/empty barrier pair, one MS*12 KB W bulk and MS x tiles per slot; the consumer
  // round must cover a whole slot (GS == MS) so a slot is released exactly once per round
  static_assert(MS == 1 || (GS == MS && !HALF), "MS > 1: GS == MS and no HALF");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XBS, bars = wring + ST * WBS;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const uint32_t ptab = bars + 24 * ST + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
  uint32_t* const ptabp = (uint32_t*)(smem + (sbase - sraw) + ptab);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  constexpr bool CUR = (FL & 1048576) != 0, NOESC = (FL & 2097152) != 0, VOTE = (FL & 4194304) != 0,
                 VASM = (FL & 8388608) != 0;
  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  // wf_rq32 FL bit 8 (ESPLIT): the entry x tiles are issued by the x lane (warp 9) in parallel with the W lane's bulks
  // (one bar.sync 6,64 after the init fence); bit 9 (EFENCE): the epilogue's fence.proxy.async.global is dropped (the
  // zeroing warp already fenced its own stores before publishing zflag).  Both measured within noise (batch 3):
  // ESPLIT -0.06..+0.11 us, EFENCE -0.22..+0.05 us on qkv m=32/16 -> no RQV entry uses them (dead code, kept for the
  // record; FL 35 is unchanged).
  constexpr bool ESPLIT = (FL & 256) != 0, EFENCE = (FL & 512) != 0;
  uint32_t ec = (warp < NW && !CUR) ? __ldg(el + lane) : 0u;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (CUR && warp < NW)  // 16 B per thread, 512 B per warp, coalesced; latency hidden behind the first TMA wait
    E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * 32 + lane) * 4);
  uint32_t hs = E4.x >> 17;
  if (ESPLIT && EARLY) {
    if (warp == PW && lane == 0) {
      for (int i = 0; i < ST; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
      }
      asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == PW || (warp == PW + 1 && XON)) asm volatile("bar.sync 6, 64;\n" ::: "memory");
    if (warp == PW && lane == 0) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, WB);
        if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
      }
    }
    if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t px = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, XB);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  } else if (tid == (EARLY ? PW * 32 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < MS * ST && s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS, slot = (s / MS) % ST;
        rq_expect(full + 8 * slot, n * WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * slot, n * XB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pw);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, px);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
      }
    }
  }
  if (tid == (PW + 2) * 32 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t ec0 = 0xffffffffu;
  if (WK && warp == PW + 3 && lane < 8) ec0 = __ldg(p.esc + (size_t)((ks * p.NB + nb) * 8 + (lane & 7)) * p.ecap);
  auto walk = [&]() {  // WK patch-table walker (warp PW+3): lane l walks math warp l's escape list and publishes
    // (hdr, 2 words per lane) into the slot's table before arriving on the slot's full barrier
    const int lw = lane & 7;
    const uint32_t* myel = p.esc + (size_t)((ks * p.NB + nb) * 8 + lw) * p.ecap;
    uint32_t* const tb0 = ptabp + lw * C::PTW;
    int ei = 0, slot = 0, r = 0;
    uint32_t ec = ec0, u0 = 0, u1 = 0;
    for (int s = 0; s < KCS; s++) {
      if (s >= ST) rq_wait((MRG ? wempty : xempty) + 8 * slot, (r & 1) ^ 1);
      uint32_t* const tb = tb0 + slot * 8 * C::PTW;
      if (lane < 8) {  // clear the words this slot's previous fill set (the consumer is done with them: xempty)
        uint32_t q0 = tb[1], q1 = tb[2];
        while (q0) { const int l = __ffs((int)q0) - 1; tb[3 + l * 2] = 0u; q0 &= q0 - 1; }
        while (q1) { const int l = __ffs((int)q1) - 1; tb[4 + l * 2] = 0u; q1 &= q1 - 1; }
        tb[0] = 0u; tb[1] = 0u; tb[2] = 0u; u0 = u1 = 0u;
      }
      while (lane < 8 && (ec >> 17) == (uint32_t)s) {
        const int lf = (int)((ec >> 12) & 31), idx = (int)((u0 >> lf) & 1);
        tb[3 + lf * 2 + idx] = 0x8000u | (((ec >> 8) & 15u) << 8) | (((ec >> 7) & 1u) << 7) | (ec & 0x7fu);
        if (idx) u1 |= 1u << lf; else u0 |= 1u << lf;
        ++ei;
        ec = __ldg(myel + ei);
      }
      if (lane < 8) { tb[1] = u0; tb[2] = u1; tb[0] = (u0 | u1) ? 1u : 0u; }
      __syncwarp();
      if (lane == 0) rq_arrive(full + 8 * slot);
      if (++slot == ST) { slot = 0; r++; }
    }
  };
  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == PW && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * WB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : (MRG ? wempty : xempty)) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * XB);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, pol);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= PW + 2) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - (PW + 2) * 32, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
      if (WK && warp == PW + 3) walk();
    } else if (WK && warp == PW + 3) {
      walk();
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  if constexpr (MREG == 232) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  else if constexpr (MREG == 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : (s / MS) % ST;
        if (!RES && (j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES && !MRG) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) rq_arrive(wempty + 8 * (((GS * G + j) / MS) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) {
          const uint32_t b = (MRG ? wempty : xempty) + 8 * (((GS * G + j) / MS) % ST);
          if (RLAX) rq_arrive_r(b); else rq_arrive(b);
        }
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          if constexpr (CURP) {
            uint32_t H[8];
            rq_hi8(Q[j][0], Q[j][1], Q[j][2], Tb, H);
            rq_fixp8(H, GS * G + j, E4, hs);
            rq_prmtA(Q[j][0], Q[j][1], H, A[j]);
          } else {
          const uint32_t* const tb = ptabp + ((((GS * G + j) / MS) % ST) * 8 + (warp & 7)) * C::PTW;
          const uint32_t ph = WK ? tb[0] : 0u, pw0 = WK ? tb[3 + lane * 2] : 0u, pw1 = WK ? tb[4 + lane * 2] : 0u;
          if (DILP) rq_unit_ilp(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          else rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          if (WK) {
            if (__builtin_expect(ph != 0u, 0)) { rq_patch(A[j], pw0); rq_patch(A[j], pw1); }
          } else if constexpr (CUR && VASM) rq_fixw(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR && VOTE) rq_fixv(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR) rq_fixc(A[j], GS * G + j, E4, hs);
          else if (NOE) { } else if (FX2) rq_fix2(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else if (WPF) rq_fixp(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          }
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : ((s / MS) % ST) * XBS + (s % MS) * XB)) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        if (!EFENCE) asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL, MS>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (info == 4) {  // wf_rq32b: CTAs per SM at this launch geometry (regs + smem)
      int nb = 0;
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, (FL & 1024) ? 256 : 384, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -1; }
      return nb;
    }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {
    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;
  }
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info) {
  RqP p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    if (esc2.numel() && esc2.numel() < nl * 128) return -14;
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.esc2 = esc2.numel() ? (const uint32_t*)esc2.data_ptr() : nullptr;
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= 2 * KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""
    def _rqwg(n):
        nd = n // 2
        d = ", ".join("%%%d" % i for i in range(nd))
        a = ", ".join("%%%d" % (nd + i) for i in range(4))
        outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
        return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
                 "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
                 "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
                % (n, nd, nd + 5, n, d, a, nd + 4, outs))
    def _rqss(n):
        nd = n // 2
        d = ", ".join("%%%d" % i for i in range(nd))
        outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
        return (("template <> __device__ __forceinline__ void rq_wgss<%d>(float (&D)[%d], uint64_t da, uint64_t db, int scd) {\n"
                 "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
                 "    : %s\n    : \"l\"(da), \"l\"(db), \"r\"(scd));\n}\n")
                % (n, nd, nd + 2, n, d, nd, nd + 1, outs))
    def _rqsrc(vs):
        cases = "\n".join("    case %d: return rq_go<%s>(p, x, out, (int)KS, (int)info);" % (i, ", ".join(str(a) for a in v[1:8] + (RQMS.get(v[0], 1),)))
                          for i, v in enumerate(vs))
        return RQCU.replace("@@WGMMA@@", _rqwg(16) + _rqwg(32) + _rqwg(128) + _rqwg(256) + _rqss(128) + _rqss(256)).replace("@@CASES@@", cases)
    RQFLAGS = ["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"]
    _rqmod = [None, None]
    def _rqext(vs=None, verbose=False, slot=None):
        vs = RQV if vs is None else vs
        src = _rqsrc(vs)
        slot = _rqmod if slot is None else slot
        if slot[0] is None or slot[1] != src:
            import os
            tag = ("rq128_%x" if slot is _rqmod else "rq32_%x") % (hash(src) & 0xffffffff)
            flags = RQFLAGS + (["-Xptxas=-v"] if verbose else [])
            slot[0] = load_inline(tag + ("v" if verbose else ""), cpp_sources="int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                                    cuda_sources=src, functions=["rqrun"], extra_cuda_cflags=flags, verbose=verbose)
            slot[1] = src
        return slot[0]
    def _pack_nat(w):
        N, K = w.shape
        W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
        return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()
    def _rqpack(w):
        N, K = w.shape
        c = _c12(_pack_nat(w), 1)
        if c is None or N % 128:
            return None
        data, tab, esc = c
        T, KC = data.shape[:2]
        lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
        return lay, tab, esc
    def _rqunpack(lay, tab, esc, N, K):
        NB, KC = lay.shape[:2]
        data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
        v = _unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
        W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
        return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)
    def _rqesc(esc6, NB, KC, KS, dev):
        G = KS * NB * 8
        if len(esc6):
            tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
            nb, wt = tile // 8, tile % 8
            s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
            ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
            u = ch - s0[ks]
            ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
            gw = (ks * NB + nb) * 8 + wt
            o = torch.argsort(gw * 65536 + u)
            gw, ent = gw[o], ent[o]
            rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
            cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
        else:
            cap = 32
        arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
        if len(esc6):
            arr[gw * cap + rank] = ent
        arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
        return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)
    def _rqesc2(esc6, NB, KC, KS, dev, CAPT=4):
        G = KS * NB * 8
        if not len(esc6):
            return torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.int32, device=dev), 0
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        slot = ((ks * NB + nb) * 8 + wt) * 32 + ln
        o = torch.argsort(slot * 65536 + u)
        slot, ent = slot[o], ent[o]
        rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
        mx = int(rank.max()) + 1
        if mx > CAPT:
            return None, mx
        arr = torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.long, device=dev)
        arr[slot * CAPT + rank] = ent
        return torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx
    RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
    RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",
              "d2s5hzwP", "s4s5hzw"]
    RQKSV = {"h128s8e": 1, "h128s6e": 1, "h128s10e": 1, "h128s8em1": 1, "h128s8eg2": 1, "h128s10eg2": 1}  # variant -> KS
    RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }
    RQBAKE = {128: [], 256: []}  # wf_jsweep: o m=128 (s4s5hz 13.06 vs r6 11.17) and all m=256 (n256* 30.8-31.9 vs r6
    RQLAYERS = ("qkv_gate", "unpooler")
    _rqthr = []
    def _rqstart(prod=None, slot=None, thr=None, tag="rq"):
        import threading
        prod = RQPROD if prod is None else prod
        thr = _rqthr if thr is None else thr

        def build():
            try:
                _rqext([RQV[RQI[nm]] for nm in prod], slot=slot)
                thr.append("ok")
                thr.append(time.time() - _T0)
            except Exception as error:
                thr.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        t = threading.Thread(target=build, daemon=True)
        t.start()
        print("PREP %s thread started at %.0fs" % (tag, time.time() - _T0), flush=True)
        return t
    def _rqcall(st, p, x, out):
        e2 = p.get("esc2")
        if e2 is None:
            e2 = st.get("rq_e2e")
            if e2 is None:
                e2 = st["rq_e2e"] = torch.empty(0, dtype=torch.int32, device=x.device)
        return p.get("mod", _rqmod[0]).rqrun(p["i"], p["lay"], x, out, p["esc"], e2, p["cnt"], p["KS"], p["tab"], 0, st["rq_e"], st["rq_sink"], 0)
    def _rqval(fn, w, m, n, gen, tol):
        K = w.shape[1]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
        exact = w[:n].t()
        if K not in _EYE:
            _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
        eye = _EYE[K]
        for start in range(0, K, m):
            x = eye[start:start + m]
            if x.shape[0] < m:
                x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=x.device)))
            if fn(x.contiguous(), out) != 0:
                return False, "launch error"
            got = out[: min(m, K - start)]
            want = exact[start:start + got.shape[0]] * 8.0
            if not torch.isfinite(got).all() or not torch.equal(got.float(), want.float()):
                return False, "one-hot mismatch at column %d" % start
        for dist in (1.0, 30.0):
            x = torch.randn((m, K), generator=gen, device=w.device)
            if dist != 1.0:
                x[:, torch.randperm(K, generator=gen, device=w.device)[: K // 100]] *= dist  # the task's 'outliers'
            x = x.to(torch.bfloat16)
            o = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
            fn(x, o)
            ref = x.double() @ w[:n].double().t()
            err = float((o.double() - ref).norm() / ref.norm())
            row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
                return False, "random rel L2 %.2e row %.2e > %.0e" % (err, row, tol)
        return True, "ok %.1e" % err
    def _rqbake(st, thr):
        st["rq"] = {}
        if time.time() - _T0 > PREPARE_BUDGET_S - 45:
            print("BAKE rq skipped-budget %.0fs" % (time.time() - _T0), flush=True)
            return
        thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 45 - (time.time() - _T0)))
        if not _rqthr or _rqthr[0] != "ok" or _rqmod[0] is None:
            print("BAKE rq build %s" % (_rqthr[0] if _rqthr else "not finished"), flush=True)
            return
        print("PREP rq thread ready at %.0fs (bake starts at %.0fs)" % (_rqthr[1], time.time() - _T0), flush=True)
        names = [v[0] for v in [RQV[RQI[nm]] for nm in RQPROD]]
        dev = st["pooler_raw"].device
        st["rq_e"] = torch.empty(0, dtype=torch.int64, device=dev)
        st["rq_sink"] = torch.zeros(4, dtype=torch.int32, device=dev)
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(12828)
        for layer in RQLAYERS:
            if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                print("BAKE rq %s skipped-budget" % layer, flush=True)
                continue
            w = st[layer + "_raw"]
            n, K = w.shape
            xs, r6a = {}, {}
            for m in (128, 256):  # r6 timed before the pack (critic F1: the first r6 timing after the pack read ~4 us high)
                xs[m] = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                o6 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6a[m] = _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[m], o6, flush, sink)
            pk = _rqpack(w)
            if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
                print("BAKE rq %s pack not exact" % layer, flush=True)
                continue
            lay, tab, esc6 = pk
            st.setdefault("rq_pk", {})[layer] = pk  # rs32: the small-m phase (after gx) reuses the pack
            KS = RQKS[layer]
            _bl = RQBAKE_L.get((layer, 128), RQBAKE[128]) + RQBAKE_L.get((layer, 256), RQBAKE[256])
            ESC = {}
            for _kk in sorted(set([KS] + [RQKSV.get(nm, KS) for nm in _bl])):
                ESC[_kk] = _rqesc(esc6, lay.shape[0], lay.shape[1], _kk, dev)[0]
            esc = ESC[KS]
            esc2, mx2 = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
            print("BAKE rq %s esc2 slots max=%d %s" % (layer, mx2, "ok" if esc2 is not None else "OVERFLOW"), flush=True)
            cnt = torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev)  # 2 per CTA, GX up to 4*NB (HALF x KS)
            _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[128], torch.empty((128, n), dtype=torch.bfloat16, device=dev), flush, sink, it=7)
            keep = False
            for m in (128, 256):
                x = xs[m]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6 = lambda xx, oo: _r6(xx, layer, n, st)
                r6med = _time_us(r6, x, out, flush, sink)
                print("BAKE %s m=%d r6 %.2f at %.0fs" % (layer, m, r6med, time.time() - _T0), flush=True)

                def _try(nm):
                    ksv = RQKSV.get(nm, KS)
                    if esc2 is None and RQI[nm] in RQCUR:
                        return None
                    p = {"i": names.index(nm), "lay": lay, "esc": ESC[ksv], "esc2": esc2, "cnt": cnt, "KS": ksv, "tab": tab, "n": n}
                    fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                    try:
                        r = fn(x, out)
                        torch.cuda.synchronize()
                        if r != 0:
                            print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, nm, r), flush=True)
                            return None
                        ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                        torch.cuda.synchronize()
                        if int(cnt.abs().sum()):  # the zero handshake must leave its counters at 0 after every launch
                            cnt.zero_()
                            ok, msg = False, "counters not reset"
                        if not ok:
                            print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, nm, msg), flush=True)
                            return None
                    except Exception as error:
                        print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, nm, str(error).splitlines()[-1][:160]), flush=True)
                        return None
                    return fn, p

                names_m = RQBAKE_L.get((layer, m), RQBAKE[m])
                pv = names_m[0] if names_m else None
                refn, refp, refmed, reffn = "r6", None, r6med, r6
                cands, done = [], set()
                if pv is not None:
                    done.add(pv)
                    got = _try(pv)
                    if got is not None:
                        fn, p = got
                        us = _time_us(fn, x, out, flush, sink, base=r6)
                        print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs r6 %.2f) at %.0fs"
                              % (layer, m, pv, us, _LAST[0][0], r6med, time.time() - _T0), flush=True)
                        rn, rm, isp = _refsel("rq:" + pv, us, "r6", r6med, RWIN128)
                        if isp:
                            refn, refp, refmed, reffn = rn, p, rm, fn
                        else:
                            cands.append(("rq:" + pv, p, _LAST[0][0]))
                for nm in names_m:
                    if nm in done:
                        continue
                    if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                        print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, nm, time.time() - _T0), flush=True)
                        continue
                    got = _try(nm)
                    if got is None:
                        continue
                    fn, p = got
                    us = _time_us(fn, x, out, flush, sink, base=reffn)
                    print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                          % (layer, m, nm, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                    cands.append(("rq:" + nm, p, _LAST[0][0]))
                best = _pickrule((refn, refp, refmed), cands, MARGIN_RQ128)
                if best[1] is not None:
                    st["rq"][(layer, m)] = best[1]
                    keep = True
                print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)
                _fo = RQPICK128.get((layer, m))
                if _fo is not None:
                    for _t, _p, _r in cands:
                        if _t == "rq:" + _fo:
                            st["rq"][(layer, m)] = _p
                            print("PICK %s m=%d rq:%s (forced: cursor + short apply, resident -0.26/-0.13/-0.32 us, bake ratio %.4f)"
                                  % (layer, m, _fo, _r), flush=True)
                            break
            if not keep:
                del lay, esc, cnt
        torch.cuda.synchronize()
    def _r6(x, layer, n, state):
        return x @ state[layer + "_raw"][:n].T


    return dict(locals())


# ---------------------------------------------------------------------------------------------- rs128 (chenwainuo)
import time as _rq_time
_CHEN = _make_chen()
_RQ_NAMES = {"qkv_gate": "qkv", "unpooler": "un"}
_rq_prior_prepare = prepare


def prepare(weights):
    t0 = _rq_time.time()
    thr = None
    try:
        thr = _CHEN["_rqstart"]()  # the rs128 extension compiles on a CPU thread while the layer builds
    except Exception as ex:
        print(f"[rq] start failed: {ex!r}"[:200], flush=True)
    st = _rq_prior_prepare(weights)
    print(f"[rq] layer prepare done at {_rq_time.time() - t0:.0f}s", flush=True)
    st["chen_rq"] = {}
    if thr is None:
        return st
    try:
        cs = {"gv": {}, "lt": {}, "pooler_raw": weights["pooler"], "qkv_gate_raw": weights["qkv_gate"],
              "unpooler_raw": weights["unpooler"]}
        _CHEN["_flushext"]()
        _CHEN["_flush_validate"]()
        _CHEN["_fixed_cost"]()
        _CHEN["_rqbake"](cs, thr)
        for (lay, m), p in cs.get("rq", {}).items():
            if (_RQ_NAMES[lay], m) in RQ_USE:
                st["chen_rq"][(_RQ_NAMES[lay], m)] = p
        st["chen_st"] = {"rq_e": cs["rq_e"], "rq_sink": cs["rq_sink"]}
        print(f"[rq] picks {sorted(st['chen_rq'])} (baked {sorted(cs.get('rq', {}))}) at {_rq_time.time() - t0:.0f}s", flush=True)
    except Exception as ex:  # the rs128 paths can only be added; the layer's own paths stand on any failure
        st["chen_rq"] = {}
        print(f"[rq] bake failed: {str(ex).splitlines()[-1][:200] if str(ex) else repr(ex)}", flush=True)
    torch.cuda.synchronize()
    return st


RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128)}


def _rq_gemm(state, name, x, T):
    p = state.get("chen_rq", {}).get((name, T)) if state is not None else None
    if p is None:
        return None
    out = torch.empty((T, p["n"]), dtype=torch.bfloat16, device=x.device)
    if _CHEN["_rqcall"](state["chen_st"], p, x.contiguous(), out) != 0:
        return None
    return out



# ---------------------------------------------------------------- MFIN: paired in-prepare timing of the MoE kernel (moe-fin, E300+)
MFAB_LAYER = True
MFAB_LAYER_SEEDS = 4
MFAB_LAYER_REPS = 24


def _mf_gen_input(bs, seed, dev):
    """Same distribution as the task's generate_input (busiest-GPU relabel); local experts are 0..31."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return x, ids.to(torch.int32).contiguous(), (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous()


def _mf_cases(sfx):
    main = tuple(_main_defines(sfx))
    return [(32, tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)), (128, main), (256, main)]


def _mf_start_base_builds(weights):
    """Compile the barrier twins (-DOKC_MFIN=0) of the builds the A/B times, on a CPU thread, while the layer prepares."""
    def run():
        try:
            sfx = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
            if not sfx:
                return
            off = ("-DOKC_SC6OFF=1",) if ("_sc6_ok" in globals() and SC6_WGS and _wgs_on() and _sc6_ok(weights["w13_sf"], weights["w2_sf"])) else ()
            for _, defs in _mf_cases(sfx):
                _ma_ext(tuple(defs) + ("-DOKC_MFIN=0",) + off + tuple(sfx))
        except Exception as ex:
            print(f"[mfab] base build failed: {str(ex)[-400:]}", flush=True)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _mf_ns_weights(st, weights):
    """The plain-scale WGS weight copies (state weights are SC6-packed when SC6 applies): the barrier twin reads these."""
    I = H = 4096
    rb = torch.arange(I // 8, device=weights["w13"].device)
    g = torch.arange(8, device=weights["w13"].device)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=weights["w13"].device)
    return st["ma_w"][:12] + [torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA), False) for e in range(32)]),
                              torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, False) for e in range(32)])]


def _mf_ab(st, t0, weights):
    dev = torch.device("cuda")
    sfx = tuple(st["ma_sfx"])
    sc6 = bool(globals().get("_MA_SC6W", False))
    w_ns = _mf_ns_weights(st, weights) if sc6 else st["ma_w"]
    if not sfx:
        print("[mfab] no PRMT scales: the MAIN build is the non-WGS fallback, nothing to compare", flush=True)
        return
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for bs, defs in _mf_cases(sfx):
        prod = _ma_ext(tuple(defs) + sfx)
        base = _ma_ext(tuple(defs) + ("-DOKC_MFIN=0",) + (("-DOKC_SC6OFF=1",) if sc6 else ()) + sfx)
        ratios, meds, maxd = [], [], 0.0
        for seed in range(MFAB_LAYER_SEEDS):
            x, ids, tw = _mf_gen_input(bs, 777 + 13 * seed + bs, dev)
            outs = {n: torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev) for n in ("base", "prod")}
            graphs = {}
            for n, ext in (("base", base), ("prod", prod)):
                def fn(n=n, ext=ext):
                    ext.okc_run(w_ns if n == "base" else st["ma_w"], st["ma_scratch"], x, ids, tw, outs[n], 0, 32, st["nsm"])
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                graphs[n] = g
            torch.cuda.synchronize()
            maxd = max(maxd, float((outs["prod"].float() - outs["base"].float()).abs().max()))
            ts = {"base": [], "prod": []}
            for r in range(MFAB_LAYER_REPS):
                for n in (("base", "prod") if r % 2 == 0 else ("prod", "base")):
                    flush.add_(1)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[n].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[n].append(e0.elapsed_time(e1) * 1000.0)
            pr = sorted(a / b for a, b in zip(ts["base"], ts["prod"]))
            ratios.append(pr[len(pr) // 2])
            meds.append(sorted(ts["prod"])[len(ts["prod"]) // 2])
        rs = sorted(ratios)
        print(f"[mfab] moe T={bs:3d} prod median {sorted(meds)[len(meds) // 2]:7.1f} us  speedup vs barrier twin ({'twin without SC6' if sc6 else 'same weights'}) {rs[len(rs) // 2]:.4f} "
              f"(per seed {' '.join(f'{v:.3f}' for v in ratios)})  max|d| {maxd:.4g}", flush=True)


_mfin_prior_prepare = prepare


def prepare(weights):
    t0 = _rq_time.time()
    th = _mf_start_base_builds(weights) if MFAB_LAYER else None
    st = _mfin_prior_prepare(weights)
    print(f"[mfab] layer prepare done at {_rq_time.time() - t0:.0f}s", flush=True)
    if MFAB_LAYER and th is not None:
        th.join(timeout=max(1.0, 195.0 - (_rq_time.time() - t0)))
        if _rq_time.time() - t0 < 205.0 and not th.is_alive():
            try:
                _mf_ab(st, t0, weights)
            except Exception as ex:  # diagnostics must never break the run
                print(f"[mfab] A/B failed: {str(ex)[-400:]}", flush=True)
        else:
            print(f"[mfab] skipped at {_rq_time.time() - t0:.0f}s", flush=True)
        torch.cuda.synchronize()
    return st


# achitaan CH: private state, synthetic inputs and error metrics from ATP (8ebcc1e).
def _ch_state(st):
    # achitaan: only weights/plans are shared. Counter/activation workspaces are private; production scratch
    # is untouched; every scheduling arm gets private counters and activations.
    ss = dict(st)
    for key in ("mm_pick", "graft_lt", "ma_ext", "chen_rq"):
        ss[key] = dict(st.get(key, {}))
    ss["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(st["qkv"].device)]
    bst = dict(st["bfp"])
    ss["bfp"] = bst
    bst["choice"] = dict(bst.get("choice", {}))
    for key in ("cnt", "mid_cnt", "diag_cnt"):
        if key in bst:
            bst[key] = {k: v.clone() for k, v in bst[key].items()}
    for key in ("dummy", "ws_shared", "mid_ws_shared", "ws_shared_diag", "fused_part", "fused_cnt"):
        if key in bst:
            bst[key] = bst[key].clone()
    if "chen_st" in ss:
        ss["chen_st"] = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in ss["chen_st"].items()}
    ss["chen_rq"] = {k: dict(v, cnt=v["cnt"].clone()) for k, v in ss["chen_rq"].items()}
    return ss


def _ch_inputs(st, B, L, sliding, seed):
    # achitaan: independent synthetic inputs following the public data contract (correlated stream, agentic
    # lengths 74000..86000, scattered page=1, soft FP8 keys); same recipe as AB32 but with a batch parameter.
    dev = st["qkv"].device
    H, per = st["qkv"].shape[1], 86000 + L
    T, total = B * L, B * per
    gen = torch.Generator(device=dev).manual_seed(seed)
    cfg = dict(batch=B, q_len=L, tokens=T, hidden=H, num_q_heads=NUM_Q_HEADS,
               num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=1,
               window=2048 if sliding else 0, rope_theta=ROPE_THETA if sliding else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0,
               n_experts=N_EXPERTS, top_k=TOP_K, local_experts=LOCAL_EXPERTS)
    data = dict(config=cfg, stream=torch.empty((T, H), device=dev),
                attn_input=torch.empty((T, H), dtype=torch.bfloat16, device=dev),
                seq_lens=torch.empty(B, dtype=torch.int32, device=dev),
                positions=torch.empty(T, dtype=torch.int64, device=dev),
                cache_loc=torch.empty(T, dtype=torch.int64, device=dev),
                cache_seqlens=torch.empty(B, dtype=torch.int32, device=dev),
                cu_seqlens_q=torch.arange(0, T + 1, L, dtype=torch.int32, device=dev),
                scheduler_metadata=None,
                page_table=torch.empty((B, per), dtype=torch.int32, device=dev),
                k_cache=torch.empty((total, 1, NUM_KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=dev),
                v_cache=torch.empty((total, 1, NUM_KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=dev),
                k_scale=torch.ones((), device=dev), v_scale=torch.ones((), device=dev),
                local_start=torch.zeros(1, dtype=torch.int32, device=dev))
    mean = torch.randn(H, generator=gen, device=dev)
    shared = torch.randn((4, H), generator=gen, device=dev)
    draft = torch.ones((B, L, 1), device=dev)
    if L == 8:
        draft[:, 1:] = 0.6
    draft = draft.view(T, 1)
    chunk = torch.empty((min(total, 1 << 18), NUM_KV_HEADS, HEAD_DIM), device=dev)

    def refresh():
        seq = torch.randint(74000, 86001, (B,), generator=gen, device=dev)
        data["seq_lens"].copy_(seq)
        data["cache_seqlens"].copy_(seq + L)
        data["page_table"].copy_(torch.randperm(total, generator=gen, device=dev).view(B, per))
        positions = seq[:, None] + torch.arange(L, device=dev)
        data["positions"].copy_(positions.flatten())
        data["cache_loc"].copy_(data["page_table"].gather(1, positions).flatten())
        coef = torch.randn((T, 4), generator=gen, device=dev)
        stream = (draft * (0.7 * mean + (0.9 / 4) ** 0.5 * (coef @ shared))
                  + (1.0 - 0.9 * draft.square()).sqrt()
                  * torch.randn((T, H), generator=gen, device=dev))
        data["stream"].copy_(stream)
        data["attn_input"].copy_(stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + 1e-5))
        u = torch.randn((NUM_KV_HEADS, HEAD_DIM), generator=gen, device=dev)
        u *= torch.rsqrt(u.square().mean(-1, keepdim=True))
        lo, hi = (0.3, 0.75) if sliding else (0.5, 0.62)
        c2 = lo + (hi - lo) * torch.rand((NUM_KV_HEADS, 1), generator=gen, device=dev)
        for key in ("k_cache", "v_cache"):
            dst = data[key].view(total, NUM_KV_HEADS, HEAD_DIM)
            for at in range(0, total, chunk.shape[0]):
                x = chunk[:min(chunk.shape[0], total - at)]
                x.normal_(generator=gen)
                if key == "k_cache":
                    x.mul_((1.0 - c2).sqrt()).add_(c2.sqrt() * u)
                dst[at:at + x.shape[0]].copy_(x)
        data["local_start"].zero_()

    return data, refresh



def _ch_cache(data):
    return tuple(data[k].view(torch.uint8).view(-1, 2, 64).index_select(0, data["cache_loc"])
                 for k in ("k_cache", "v_cache"))


def _ch_reset_cache(data, before):
    for key, value in zip(("k_cache", "v_cache"), before):
        data[key].view(torch.uint8).view(-1, 2, 64).index_copy_(0, data["cache_loc"], value)


def _ch_error(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf"), float("inf")
    a, b = a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1)
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return float("inf"), float("inf")
    diff = torch.linalg.vector_norm(a - b, dim=-1)
    norms = torch.linalg.vector_norm(b, dim=-1)
    floor = (norms.square().mean().sqrt() * 0.1).clamp_min(1e-20)
    return (float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(norms).clamp_min(1e-20)),
            float((diff / norms.clamp_min(floor)).max()))


# achitaan CH: scheduling-only arms. No weight, arithmetic, precision or backend
# changes. The original decode_v1 definitions above remain the fallback.
# A: row-boundary operands before wait + early launch; T=16 keeps its EXACT
# cluster arithmetic and already-preloaded operands, only moving the signal.
# B: the prepared rs128 winner, with W-ring bootstrap before wait and a PDL
# launch. C: permit the existing _pdl-controlled stages at T=256 as well.
import hashlib as _ch_hashlib
import statistics as _ch_stats
import time as _ch_time

_CH_PARENT_PREPARE = prepare
_CH_PARENT_KERNEL = kernel_fn
_CH_OLD_BOUNDARY = _graft_boundary
_CH_OLD_RQ = _rq_gemm
_CH_OLD_DENSE = dense
_CH_OLD_POOL = _pool_router_fp32
_CH_OLD_ATTENTION = _decode_attention7
_CH_TLS = threading.local()
_CH_SHAPES = tuple((B, L, s) for L in (8, 1) for B in (16, 32) for s in (True, False))
_CH_ARMS = ("A", "A+B", "A+B+C")
_CH_BLOCKS = 12  # 24 replays of EACH arm per draw, three fresh draws


def _pdl(T):
    # Thread-local launch policy: no mutation of PDL_MAX_T across shape classes.
    return PDL and T <= (256 if "C" in getattr(_CH_TLS, "arm", "base") else PDL_MAX_T)


@_glx.jit
def _ch_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                     H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr,
                     V: _gl.constexpr, NW: _gl.constexpr):
    # achitaan: same expressions/layout/casts as _graft_row_boundary_pre.
    # Stream is an older residual, not the immediate projection's output. For
    # boundary 2, MoE's post-wait launch signal orders the boundary-1 residual.
    _gdc_launch()
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    s = _gl.load(Stream + row * H + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    h = ns * r * pw
    _gl.store(Out + row * H + c, h)


def _ch_mark(name, value):
    obs = getattr(_CH_TLS, "observe", None)
    if obs is not None:
        obs[name] = value  # retain pointers only: no diagnostic GPU nodes


def _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):
    arm = getattr(_CH_TLS, "arm", "base")
    if "A" in arm and S == 1 and T in (16, 32, 128, 256) and P.shape[-1] == HIDDEN:
        g, br = st["graft"], P.view(T, HIDDEN)
        _graft_al(br, stream, s_out, h_out)
        if T == 16 and br.dtype == torch.bfloat16 and (not post or nu == HIDDEN):
            ext = st.get("ch_cluster")
            if ext is None:
                raise RuntimeError("CH cluster extension unavailable")
            if post:
                ext.b2c(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            else:
                ext.b1(br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out,
                       router_out if router_out is not None else s_out, router_out is not None)
            _ch_mark("used_A", True)
            return True
        if T > 16:
            pfx, nw = ("2" if post else "1"), (16 if T > 192 else 32)
            _ch_row_boundary[(T,)](br, stream, g["post2"], g["pre" + pfx], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, HIDDEN, post, 4, nw,
                                  num_warps=nw, launch_pdl=True)
            _ch_mark("used_A", True)
            return True
    return _CH_OLD_BOUNDARY(P, S, T, stream, st, post, s_out, h_out, nu, router_out)


def _decode_attention7(data, state, qkv):
    out = _CH_OLD_ATTENTION(data, state, qkv)
    _ch_mark("qkv", qkv)
    _ch_mark("attn", out)
    return out


def dense(x, w, T, state=None, name=None):
    result = _CH_OLD_DENSE(x, w, T, state, name)
    if name in ("qkv", "o", "pr"):
        _ch_mark("proj_" + name, result[0])
    return result


def _pool_router_fp32(h, hr, T, state, ppick):
    result = _CH_OLD_POOL(h, hr, T, state, ppick)
    _ch_mark("preffn", h)
    p, splits = result
    if splits == -1:
        _ch_mark("pooled", p[0])
        _ch_mark("logits", p[1])
    else:
        _ch_mark("pool_router", p)
    return result


def _rq_gemm(state, name, x, T):
    if "B" not in getattr(_CH_TLS, "arm", "base"):
        return _CH_OLD_RQ(state, name, x, T)
    p = state.get("chen_rq", {}).get((name, T)) if state is not None else None
    if p is None:
        return None
    ext = state.get("ch_rq_ext")
    indices = state.get("ch_rq_indices", {})
    if ext is None or p["i"] not in indices:
        raise RuntimeError("CH rq extension/variant unavailable (no timed-path JIT)")
    alt = dict(p, i=indices[p["i"]], mod=ext)
    out = torch.empty((T, p["n"]), dtype=torch.bfloat16, device=x.device)
    rc = _CHEN["_rqcall"](state["chen_st"], alt, x.contiguous(), out)
    if rc != 0:
        raise RuntimeError(f"CH rq launch status={rc}")
    _ch_mark("used_B", True)
    return out


def _ch_replace(src, old, new):
    if src.count(old) != 1:
        raise RuntimeError(f"CH source anchor count={src.count(old)}: {old[:70]}")
    return src.replace(old, new)


def _ch_rq_source(vs):
    # achitaan: specialize ONLY the variants already selected by base prepare.
    # Both the EARLY and non-EARLY originals use the same ring protocol; this
    # bootstrap issues all initial W chunks, waits, then issues their X tiles.
    # Full-barrier arrivals and all math/reduce-add code are unchanged.
    for v in vs:
        if v[1] not in (128, 256) or v[4] != 0 or v[6] != 0 or v[7] & (16 | 256 | 1024):
            raise RuntimeError(f"CH unsupported rq geometry {v}")
    src = _CHEN["_rqsrc"](vs)
    src = _ch_replace(src, "constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;",
                      "constexpr bool EARLY = true; // achitaan CH: bootstrap both rings below")
    a = src.index("  if (ESPLIT && EARLY) {")
    b = src.index("  if (tid == (PW + 2) * 32 && XON)", a)
    src = src[:a] + r'''
  // achitaan CH: weights/escape lists are independent of the prior grid.
  if (tid == PW * 32) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8*i), "r"(2 + (WK ? 1 : 0)));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8*i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8*i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    const int nbw = p.walias ? nb % p.walias : nb;
    const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288;
    uint64_t pw = 0;
    if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
    for (int s = 0; s < MS * ST && s < KCS; s += MS) {
      const int n = (KCS - s < MS) ? KCS - s : MS, slot = s / MS;
      rq_expect(full + 8 * slot, n * WB);
      if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pw);
      else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
    }
  }
  __syncthreads();
  // Every thread waits before X reads OR output/counter reuse. In particular,
  // zeroing must not race a prior invocation using the same persistent counters.
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  if (tid == (PW + 1) * 32) {
    uint64_t px = 0;
    if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
    for (int s = 0; s < MS * ST && s < KCS; s += MS) {
      const int n = (KCS - s < MS) ? KCS - s : MS, slot = s / MS;
      rq_expect(full + 8 * slot, n * XB);
      for (int j = 0; j < n; j++) {
        if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0+s+j)*64, 0, full+8*slot, px);
        else rq_tile(xring + slot * XBS + j * XB, &tx, (s0+s+j)*64, 0, full+8*slot);
      }
    }
  }
''' + src[b:]
    src = _ch_replace(src, "    return;\n  }\n  // ---------------- math warpgroups:",
                      '    // achitaan CH: the W producer has issued its last ring stage.\n'
                      '    if (warp == PW && lane == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
                      '    return;\n  }\n  // ---------------- math warpgroups:')
    old = "  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);"
    src = _ch_replace(src, old, r'''
  // achitaan CH: preserve grid/block/smem, add programmatic serialization.
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(p.NB * ((FL & 16) ? 2 : 1), KS);
  cfg.blockDim = dim3((FL & 1024) ? 256 : 384);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr; cfg.numAttrs = 1;
  const cudaError_t launch = cudaLaunchKernelEx(&cfg, k, p, tx, to);
  if (launch != cudaSuccess) return (int)launch;
''')
    return src


def _ch_start_builds(st):
    # Build results go into a private mailbox, never a live production state.
    results, threads = {}, []
    device = st["qkv"].device

    def build(kind):
        try:
            torch.cuda.set_device(device)
            if kind == "cluster":
                from tvm_ffi.cpp import load_inline as tvm_load
                src = _ch_replace(_GRAFT_GLUE_CUDA,
                                  '  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");',
                                  '  asm volatile("griddepcontrol.wait;" ::: "memory"); // achitaan CH')
                src = _ch_replace(src, "  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N];",
                                  '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); // achitaan CH\n'
                                  '  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N];')
                tag = _ch_hashlib.sha256(src.encode()).hexdigest()[:12]
                ext = tvm_load(name="achitaan_ch_cluster_" + tag, cuda_sources=src, functions=["b1", "b2c"],
                               extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--fmad=false",
                                                  "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])
                results["ch_cluster"] = _RouterExt(ext)  # achitaan: preserve the base state's deepcopy contract
            else:
                indices = sorted({p["i"] for (_, T), p in st.get("chen_rq", {}).items() if T in (128, 256)})
                vs = [_CHEN["RQV"][_CHEN["RQI"][_CHEN["RQPROD"][i]]] for i in indices]
                src = _ch_rq_source(vs)
                tag = _ch_hashlib.sha256(src.encode()).hexdigest()[:12]
                cpp = ("int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                       "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);")
                ext = load_inline("achitaan_ch_rq_" + tag, cpp_sources=cpp, cuda_sources=src,
                                  functions=["rqrun"], extra_cuda_cflags=_CHEN["RQFLAGS"], verbose=False)
                results.update(ch_rq_ext=_RouterExt(ext), ch_rq_indices={old: new for new, old in enumerate(indices)})
                print(f"CH_INFO rq_variants={','.join(v[0] for v in vs)}", flush=True)
        except Exception as ex:
            print(f"CH_ERR build={kind} {type(ex).__name__}: {str(ex)[-400:]} fallback=base", flush=True)

    kinds = []
    if "graft" in st and st.get("compose_graft_enabled", {}).get(16, False):
        kinds.append("cluster")
    if st.get("chen_rq"):
        kinds.append("rq")
    for kind in kinds:
        th = threading.Thread(target=build, args=(kind,), daemon=True)
        threads.append(th)
        th.start()
    return results, threads


def _ch_key(cfg):
    return cfg["batch"], cfg["q_len"], bool(cfg["window"])


def _ch_compatible(data):
    # achitaan: do not extrapolate admission to other windows/scales/strides.
    c, x = data["config"], data["attn_input"]
    return (_ch_key(c) in _CH_SHAPES and c["tokens"] == c["batch"] * c["q_len"]
            and c["page_size"] == 1 and c["window"] in (0, 2048)
            and c["rope_theta"] == (ROPE_THETA if c["window"] else 0.0)
            and c["num_q_heads"] == 16 and c["num_kv_heads"] == 2 and c["head_dim"] == 64
            and c.get("kv_scale") == 1.0 and c["qk_scale_factor"] == 5.4730064863875 and c["sm_scale"] == 0.125
            and x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (c["tokens"], HIDDEN)
            and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()
            and all(data[k].dtype == torch.float8_e4m3fn and data[k].is_contiguous() for k in ("k_cache", "v_cache")))


def kernel_fn(data, state):
    key = _ch_key(data["config"])
    arm = state.get("ch_pick", {}).get(key, "base")
    if arm != "base" and not _ch_compatible(data):
        arm = "base"
    saved = getattr(_CH_TLS, "arm", "base"), getattr(_CH_TLS, "observe", None)
    _CH_TLS.arm, _CH_TLS.observe = arm, state.get("_ch_observe")
    try:
        return _CH_PARENT_KERNEL(data, state)
    except Exception as ex:
        if arm == "base":
            raise
        state.setdefault("ch_pick", {}).pop(key, None)
        state["_ch_failed"] = True
        print(f"CH_ERR shape={key} arm={arm} {type(ex).__name__}: {str(ex)[-200:]} fallback=base", flush=True)
        _CH_TLS.arm = "base"
        return _CH_PARENT_KERNEL(data, state)
    finally:
        _CH_TLS.arm, _CH_TLS.observe = saved


def _ch_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _ch_values(arm, carry):
    vals = {k: v.reshape(-1, v.shape[-1]) for k, v in arm["state"]["_ch_observe"].items()
            if isinstance(v, torch.Tensor)}
    vals["hidden"] = arm["output"][0]
    # achitaan: remove BOTH known residual gates so carry cannot hide bad MoE.
    vals["update"] = arm["output"][1] - carry
    return vals


def _ch_id_error(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.ndim != 2 or a.shape[1] != TOP_K:
        return float("inf")
    # Set distance, not rank swaps. IDs must still be valid and unique per row.
    if not bool(((a >= 0) & (a < N_EXPERTS)).all()):
        return float("inf")
    aa = a.sort(-1).values
    if bool((aa[:, 1:] == aa[:, :-1]).any()):
        return float("inf")
    return float((~(a[:, :, None] == b[:, None, :]).any(-1)).float().mean())


def _ch_calibrate(base, data, before, carry, flush, sink, label, draw):
    # achitaan: reference = two UNMODIFIED decode_v1 graph executions on SAME
    # inputs/cache bytes. Not a mathematical golden; official conformance is
    # still required. The default's nondeterministic atomic reductions/router
    # near-ties must not be mistaken for a scheduling regression.
    refs = []
    for i in range(2):
        _ch_reset_cache(data, before)
        torch.sum(flush, dim=0, out=sink)
        base["graph"].replay()
        torch.cuda.synchronize()
        vals = {k: v.clone() for k, v in _ch_values(base, carry).items()}
        ids = base["output"][2].clone()
        if i == 0:
            cache = tuple(data[k].view(torch.uint8).clone() for k in ("k_cache", "v_cache"))
            cache_slots = _ch_cache(data)
        else:
            second_slots = _ch_cache(data)
        refs.append((vals, ids))
    if refs[0][0].keys() != refs[1][0].keys() or not {"qkv", "attn", "proj_o", "preffn"} <= refs[0][0].keys():
        raise RuntimeError("CH incomplete native pre-MoE observations")
    first, second = refs[0][0], refs[1][0]
    spread = {k: _ch_error(second[k], v) for k, v in first.items()}
    exact = {k for k, v in first.items() if k not in ("hidden", "update") and _ch_bits(second[k], v)}
    idspread = _ch_id_error(refs[1][1], refs[0][1])
    cache_repeat = all(torch.equal(data[k].view(torch.uint8), v) for k, v in zip(("k_cache", "v_cache"), cache))
    # achitaan: if a split-K qkv near a FP8 rounding boundary makes the default's
    # NEW cache slots nonrepeatable, calibrate just those slots. Every untouched
    # cache byte must still match, without tolerance or hashing.
    kv_spread = tuple(_ch_error(b.view(torch.float8_e4m3fn), a.view(torch.float8_e4m3fn))
                      for a, b in zip(cache_slots, second_slots))
    kv_exact = tuple(torch.equal(a, b) for a, b in zip(cache_slots, second_slots))
    _ch_reset_cache(data, cache_slots)
    untouched_ok = all(torch.equal(data[k].view(torch.uint8), v) for k, v in zip(("k_cache", "v_cache"), cache))
    _ch_reset_cache(data, second_slots)
    finite = all(e < float("inf") for es in spread.values() for e in es) and idspread < float("inf")
    finite &= all(e < float("inf") for es in kv_spread for e in es)
    report = ";".join(f"{k}:{e[0]:.2e}/{e[1]:.2e}" for k, e in sorted(spread.items()))
    print(f"CH_INFO class={label} draw={draw} compare=two_base_runs+full_KV_bytes+preMoE+hidden+carry_removed_update+ID_sets "
          f"exact_pre={','.join(sorted(exact))} base_KV_exact={int(cache_repeat)} base_ids={idspread:.3e} "
          f"base_spread_rel/row={report} KV_new_spread={kv_spread} untouched_KV_exact={int(untouched_ok)} "
          "tolerance=max(1e-3/5e-3,3xspread)", flush=True)
    return dict(values=first, ids=refs[0][1], cache=cache, spread=spread, exact=exact,
                idspread=idspread, valid=finite and untouched_ok, cache_slots=cache_slots,
                kv_spread=kv_spread, kv_exact=kv_exact)


def _ch_equivalence(arm, data, ref, carry):
    obs, out = arm["state"]["_ch_observe"], arm["output"]
    valid = ref["valid"] and len(out) == 3 and not arm["state"].get("_ch_failed", False)
    vals = _ch_values(arm, carry)
    if vals.keys() != ref["values"].keys():
        return False, False, False, float("inf"), float("inf")
    exact_ok, worst = True, 0.0
    for k, value in vals.items():
        error = _ch_error(value, ref["values"][k])
        if k in ref["exact"]:
            equal = _ch_bits(value, ref["values"][k])
            exact_ok &= equal
            valid &= equal
        for e, floor, spread in zip(error, (1e-3, 5e-3), ref["spread"][k]):
            limit = max(floor, 3.0 * spread)
            valid &= e <= limit and e < float("inf")
            worst = max(worst, e / limit)
    ids = _ch_id_error(out[2], ref["ids"])
    valid &= ids <= 3.0 * ref["idspread"] and ids < float("inf")
    # achitaan: deterministic K/V are always bit-exact. Nondeterministic new
    # slots get the same calibrated numerical test, not a global byte-error
    # allowance that could conceal a write to an unrelated page.
    slots = _ch_cache(data)
    cache = True
    for got, expected, exact, spread in zip(slots, ref["cache_slots"], ref["kv_exact"], ref["kv_spread"]):
        if exact:
            cache &= torch.equal(got, expected)
        else:
            errors = _ch_error(got.view(torch.float8_e4m3fn), expected.view(torch.float8_e4m3fn))
            cache &= all(e <= max(f, 3.0 * s) and e < float("inf")
                         for e, f, s in zip(errors, (1e-3, 5e-3), spread))
    if all(ref["kv_exact"]):
        cache &= all(torch.equal(data[k].view(torch.uint8), v) for k, v in zip(("k_cache", "v_cache"), ref["cache"]))
    else:
        _ch_reset_cache(data, ref["cache_slots"])
        try:
            cache &= all(torch.equal(data[k].view(torch.uint8), v) for k, v in zip(("k_cache", "v_cache"), ref["cache"]))
        finally:
            _ch_reset_cache(data, slots)
    valid &= cache
    if arm["name"] != "base":
        valid &= obs.get("used_A", False)
        if "B" in arm["name"]:
            valid &= obs.get("used_B", False)
    return bool(valid), bool(exact_ok), cache, worst, ids


def _ch_capture(data, st, key, name):
    ss = _ch_state(st)
    ss["ch_pick"] = {} if name == "base" else {key: name}
    ss["_ch_observe"] = {}
    ss.pop("_ch_failed", None)
    stream = torch.cuda.Stream(device=st["qkv"].device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            kernel_fn(data, ss)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = kernel_fn(data, ss)
    graph.replay()
    torch.cuda.synchronize()
    if ss.get("_ch_failed"):
        raise RuntimeError(f"CH capture fell back arm={name}")
    if name != "base" and (not ss["_ch_observe"].get("used_A") or ("B" in name and not ss["_ch_observe"].get("used_B"))):
        raise RuntimeError(f"CH requested arm was not reached: {name}")
    return dict(name=name, state=ss, graph=graph, output=output, old_times=[], times=[], pairs=[],
                valid=True, exact=True, cache=True, worst=0.0, iderror=0.0, draws=0, done=0)


def _ch_shape(st, key, flush, sink, deadline):
    B, L, sliding = key
    T = B * L
    label = f"{'d' if L == 1 else 'v'}{B}{'s' if sliding else 'f'}"
    names = ["A"]
    if T in (128, 256) and st.get("ch_rq_ext") is not None and any(t == T for _, t in st.get("chen_rq", {})):
        names.append("A+B")
        if T == 256:
            names.append("A+B+C")
    if T == 16 and st.get("ch_cluster") is None:
        print(f"CH_ERR class={label} arm=A cluster_unavailable_or_base_path_not_grafted fallback=base", flush=True)
        return None
    print(f"CH_INFO class={label} arms=base,{','.join(names)} "
          f"omitted_aliases={'A+B+C=A+B' if T == 128 else 'B,C_no_effect' if T <= 32 else 'none'}", flush=True)
    data, refresh = _ch_inputs(st, B, L, sliding, 20261102 + B + T + 1000 * int(sliding))
    refresh()
    if not st["att_ok"] or not _supported7(data["config"]) or not _att_fits7(data, data["config"]):
        raise RuntimeError("CH native baseline attention unavailable")
    base = _ch_capture(data, st, key, "base")
    arms = []
    for name in names:
        if _ch_time.monotonic() > deadline - 8.0:
            break
        arms.append(_ch_capture(data, st, key, name))
    g1, g2 = st["gate1"].float() / 0.3, st["gate2"].float() / 0.3
    alpha = (torch.sigmoid(-g1) * (1 + torch.sigmoid(g1))).clamp_min(1e-3).sqrt()
    alpha *= (torch.sigmoid(-g2) * (1 + torch.sigmoid(g2))).clamp_min(1e-3).sqrt()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for draw in range(3):
        if not arms or _ch_time.monotonic() > deadline - 5.0:
            break
        refresh()
        before = _ch_cache(data)
        base["graph"].replay()
        ids = base["output"][2]
        busiest = int(torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=8).argmax())
        data["local_start"].fill_(busiest * LOCAL_EXPERTS)
        carry = data["stream"] * alpha
        ref = _ch_calibrate(base, data, before, carry, flush, sink, label, draw)
        for arm in arms:
            arm["draws"] += 1
        for block in range(_CH_BLOCKS):
            if _ch_time.monotonic() > deadline - 3.0:
                break
            # Rotate candidate order as well as ABBA/BAAB to avoid always timing
            # C last. Every candidate has its OWN contemporaneous base timings.
            ordered = arms if (block + draw) % 2 == 0 else list(reversed(arms))
            for arm in ordered:
                bt = [[], []]
                order = (0, 1, 1, 0) if block % 2 == 0 else (1, 0, 0, 1)
                for i in order:
                    run = base if i == 0 else arm
                    _ch_reset_cache(data, before)
                    torch.sum(flush, dim=0, out=sink)  # 512 MiB READ, before EVERY replay
                    start.record()
                    run["graph"].replay()
                    end.record()
                    end.synchronize()
                    us = start.elapsed_time(end) * 1000.0
                    bt[i].append(us)
                    (arm["old_times"] if i == 0 else arm["times"]).append(us)
                    ok, exact, cache, worst, iderr = _ch_equivalence(run, data, ref, carry)
                    arm["valid"] &= ok  # default repeats must also stay within its measured spread
                    arm["exact"] &= exact
                    arm["cache"] &= cache
                    arm["worst"] = max(arm["worst"], worst)
                    arm["iderror"] = max(arm["iderror"], iderr)
                arm["pairs"].append(sum(bt[1]) / sum(bt[0]))
                arm["done"] += 2
        del ref
    torch.cuda.synchronize()
    _check_guards()
    admitted = []
    for arm in arms:
        arm["old_us"] = _ch_stats.median(arm["old_times"]) if arm["done"] else float("nan")
        arm["new_us"] = _ch_stats.median(arm["times"]) if arm["done"] else float("nan")
        arm["ratio"] = arm["new_us"] / arm["old_us"]
        arm["paired"] = _ch_stats.median(arm["pairs"]) if arm["pairs"] else float("nan")
        arm["eligible"] = (arm["valid"] and arm["done"] == 72 and arm["draws"] == 3
                           and arm["ratio"] <= 0.99 and arm["paired"] <= 0.99
                           and abs(arm["ratio"] - arm["paired"]) <= 0.01)
        if arm["eligible"]:
            admitted.append(arm)
    best = min(admitted, key=lambda a: max(a["ratio"], a["paired"])) if admitted else None
    for arm in arms:
        print(f"CH class={label} arm={arm['name']} old_us={arm['old_us']:.3f} new_us={arm['new_us']:.3f} "
              f"ratio={arm['ratio']:.5f} paired={arm['paired']:.5f} equiv={int(arm['valid'])} "
              f"adopted={int(arm is best)} eligible={int(arm['eligible'])} exact_pre={int(arm['exact'])} "
              f"KV_untouched+det_exact={int(arm['cache'])} max_error/tol={arm['worst']:.3f} id_error={arm['iderror']:.3e} "
              f"reps={arm['done']} draws={arm['draws']}", flush=True)
    for name in names[len(arms):]:
        print(f"CH class={label} arm={name} old_us=nan new_us=nan ratio=nan paired=nan equiv=0 adopted=0 budget=1", flush=True)
    return best["name"] if best is not None else None


def _ch_run(st, prepare_start):
    global _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR
    deadline = prepare_start + 280.0  # reserve 20 s for cleanup; never admit an incomplete trial
    picks = {}
    saved = (_WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR)
    _WS7, _CL7, _GUARDS, _GUARDED_IDS = {}, {}, [], set()
    try:
        flush = torch.ones((512 << 20) // 4, dtype=torch.float32, device=st["qkv"].device)
        sink = torch.empty((), dtype=torch.float32, device=st["qkv"].device)
        for key in _CH_SHAPES:
            if _ch_time.monotonic() > deadline - 10.0:
                print(f"CH_ERR class={key} budget_exhausted fallback=base", flush=True)
                continue
            try:
                pick = _ch_shape(st, key, flush, sink, deadline)
                if pick is not None:
                    picks[key] = pick
            except Exception as ex:
                print(f"CH_ERR class={key} {type(ex).__name__}: {str(ex)[-400:]} fallback=base", flush=True)
                print(f"CH class={key} arm=all old_us=nan new_us=nan ratio=nan paired=nan equiv=0 adopted=0", flush=True)
                # Device faults cannot be repaired by a Python fallback. Fail
                # loudly on a poisoned context, rather than admit other shapes.
                torch.cuda.synchronize()
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            (_WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR) = saved
            bfp__get_ext().sk2_set_pdl(bfp_SK2_PDL)
            for ext in st["ma_ext"].values():
                ext.okc_set_pdl(0)
    return picks


def prepare(weights):
    global MFAB_LAYER
    start, old_mfab = _ch_time.monotonic(), MFAB_LAYER
    try:
        MFAB_LAYER = False  # achitaan: skip old non-selecting write-flush diagnostic ONLY
        st = _CH_PARENT_PREPARE(weights)
    finally:
        MFAB_LAYER = old_mfab
    st["ch_pick"] = {}
    try:
        elapsed = _ch_time.monotonic() - start
        print(f"CH_INFO base_prepare_s={elapsed:.1f} baseline=decode_v1 flush_bytes=536870912 "
              "timing=full_layer_graph_raw_events draws=3 reps_per_draw=24 arms=base,A,A+B,A+B+C", flush=True)
        if st.get("bfp") is None or elapsed > 240.0 or torch.cuda.get_device_capability(st["qkv"].device)[0] != 9:
            print("CH_ERR no_gate_budget_or_backend fallback=base_all", flush=True)
        else:
            if elapsed < 210.0:
                mailbox, threads = _ch_start_builds(st)
                for th in threads:
                    th.join(timeout=max(0.0, start + 225.0 - _ch_time.monotonic()))
                st.update(dict(mailbox))  # late build results cannot change an admitted state
            with torch.no_grad():
                picks = _ch_run(st, start)
                _check_guards()
            st["ch_pick"] = picks  # atomic admission only after cleanup and guard checks
    except Exception as ex:
        st["ch_pick"] = {}
        print(f"CH_ERR gate_aborted {type(ex).__name__}: {str(ex)[-400:]} fallback=base_all", flush=True)
    print(f"CH_INFO total_prepare_s={_ch_time.monotonic() - start:.1f} adopted={len(st['ch_pick'])}/8", flush=True)
    return st

