# okc: problem=muse_prefill_layer
"""DIAGNOSTIC ONLY: deliberately wrong outputs; read MD lines, not test results.

The frozen incumbent (including its license) is embedded below, not imported
from the filesystem. All experiments run in prepare; kernel_fn returns zeros.
Only the four builds selected by M=1,8,16,32,128,256 are compiled, four variants
each. BASE device code is unchanged. Macro edits are exact-anchor checked in
diagnostic_source(). NODQ is an ablation, NOT a mathematically valid speedup.

Timing: 24 graph/event repetitions, 256 MiB L2 write flush outside the events,
raw median/min microseconds (NO judge graph-overhead subtraction). MB=1e6 B;
TBps is touched-weight bytes / median, NOT measured physical HBM traffic.
NOB still reads zero-initialized ybuf in the unmodified finalizer. NOFIN keeps
the grid barrier and scratch reset; skinny norm statistics in B remain.
NODQ keeps TMA extents and consumes packed codes AND scales via volatile XOR;
register allocation/scheduling and NaN arithmetic can change: compare resources.
STREAM is a contiguous equal-byte floor, not the MoE's repeated-job traffic.
EMPTY uses real zero-local routing for WS; MAIN has no nj==0 guard in
wgs_stage_at, so uses a trivial launch of the same grid/block geometry.

Case definitions: TASK.md loads/weights plus recorded public job f0f01b3229f94cbb
case seeds (task.yml is not shipped here). Dist omitted there means normal;
rank=busiest, ep8. These are representative fixed routings, NOT the judge's
100 fresh routings. generate_input is reproduced from reference.py; weights
use nvfp4.random_weights' formula with leading=(32,) instead of (256,) once.
"""

import base64
import hashlib
import math
import os
import queue
import statistics
import threading
import time
import types
import zlib

import torch

INCUMBENT_B85 = "c-riJeOntxvhV+VidputGh>9TkuWC6I2UYVb8KT@3m@-3EGv(skr0THn2`{~$@%QJepTH)Jul+LcCvfVI!_WbZ`Iw^)z$Uhe|5&E-A!i_k2|CAZ0l^6gzd9&*q*Ni{`%5io#$aZya;<vKOUbs|9uk9E-TLQbUH~kA3humr^mBSwL3n0c!4Jm3Dj!6R>ybmhfzeI#o@&;8IB{T-*^<P`C;gH{YKdE*Ly*y(|`Q9zR{@l)*g52jn3MmM&sv3)7hQ`-Q#ezUaeIL&YN*J7&+bJAR2_pCc!%!UxZF?7!0CuG97l4Cf!S><2ZEQ2JvVZIXk=ly7ThI+qVw~qj4t~ttJE^Bv764Eb0aEFie~v>dAXyzaMs|G(smD$7jww91f1B6#_X8hS4w@IJ@r~b$Lc#PLIz*Y7_lx9S1RuFX_g?w0le+oX>{T5htM^VKg05r{i!k3a+Zo?rb!2M#F9xC85(Fk4EEp0uVR;BbadB$fwtT@!w(0cqX(}oc{`<SrA`2^;&K1n$IwgG>!B5yc%#*)p0y{sCt?_q%Y;;gBKs(ww&$#=gzZt`_K0d_TKHcoZWXHoR6&+73af?_aELp|M(2IEBw~;z1G2py`7J^V}SToqNP3|bxPgj@tdA4s}U|c$#FoqItzoynbOFoVSFZ&)g4DY`I0agJ4B_JB%<2htc%+f0}OBXhRHM@c4m0u1c}o_!!%o+E2kB91&mKE#N*lE*!kHR_vvCt?~S{&Gn&1!-i_lz_qyZBRSbf0#`7qQ9qJgSetP8uv+40T9{$R08c1c|Opk-9Lro9jfIcOHo2E0K>qa;rqI$s%6uL5tFjO8#=mcGcr29e$p?3|SF}<QXilx9EAoNb-@u)&X9_ot`cfNvA;V!Km`i$24**G$=)T_>%mWDu*Mv3UhG52LMizj2ErWD792+}!QR&bViK#AiHJ^6S%5910=XH2t(jtnFDU2&#kryCF%;5`FJ9x^6k=PZbVL5S%@+sTYpsdTR5%#TAx0?`Av95D1OhUP=i7{PIe)El0x<ajs%tp2c1c%6iC7r@;|wR;{n2qq*`kb;^`lPS#r<|6^iNHk0W+6f~<Mi;yy1K;XvmIVGjo-I2reUD%9vS(8P{R`||40|)ciyfO@42bZOdQBXex<max8zu=a2Btn)AUsK>1lXcByR=CA;D$N+o5W#1jKLB4X`k_bie}Eny&(-U;8mAS`7r8^W{ehEro>>U#39dy=mX7cGVV|3Ai9K`q)F=0T<F!wpbQ{+rJ}h+e>j-M{02?aD9kaMcbyX=km3-6=t}O<1kFagdU(S-3$gQ1@6+W&)Dk6lH_(K}cSh>6?*xt@iUC%#12s@`1Eys)8DhbWxr1uJ1ELQa3*F8PJ<IE8%ooD@6F`#HeHPLV=}fOCc?5hK$EOA6F{hi{bLLv0!gPtyhN^~REf7JY8sJ&bBbIX!3`YTp5Q<M&T&w~P4ASfdio^noe`sz<2%*;vI~1(Z>qACyFr8w5VqEDS8z3$XBfJdGCe%88N}B>L0{Im0Zcip5wSGy<jdoryiw_oW0u}&(nq)ao6=<m(8wv{smS95<pc5pR&WM+5&k~}9v!OAbKCG1a@vwWGVH-4kB+`fwAp%qa3MV9~b>xJE?QyKH2#A_^d$k#G*eSz==!fSbpa#byUNZFIa4?Lhogxhsd3egz`WZG+aq?(XSOk@;ndAo4ZbL#GNmuC#4ktm(v;+L`V9vrg99_{0iB1`T9ikslP86Jlo+g4Jv8=csba`{BWOiAD-a^NKrEuKOQo%E90!m&erGWhEFpE0VszIh&ka|Bhy#ct)3^8r@lw^~_8VjEIJ9=Z$AS%{!nPR^fQ{$tQ*(b9OaTvv7v=qY>$NlFnszt(0G50TsDCw;8hI#EiV+%gu<J<wPf=)<C?Gr+-6NYXlH8{&=T+0TU$PI>#(`Vs`7H2#r=3T*jkRrg8HIMOS#DdH$QuE+oZDi3OrU(PRrb)VpcqWp{wOg3Q;p}$k-?U%EIfo;BJ|c0I;8~lU5v-D{WE!3&S(Z<`_AJD<)8!pjy(7ecT~j0<BfZ(^uVmOvh9)e$1Ain#cV`Jp3f$%y^Hn9Feqyd@Z)oAAMn-nTnnGzn$z<4_jb{lh?6V*~1$&HBSt}Ld!elszc&{Mh#C$P|OKB7AY<ZtBA2{|Jt}d4^@0<*8hOgJ=Ed(eVnc$;mxu%_C>QW~p3L{Z8WUfzLw_CE8r_vTjXu3Nl+{<<kaoSp#DYL3qotF^R(D*Zh3N5WUt(k1hih@hkr_2(xd3Z=WtYe|d0h6J=vuwz+1qq!rT9O+l;goQc5+$+7QExtk%o2^GRi=f6#*WKXlAz)N)V1+dFq&Si_T!MQh9oRrkZ#`@g(cE})|RPJt>w`-wEiZb?*c!x+5cqLp|1&}M8MF$E9o8Txopx2-%-MetuJQP3B$KpATP99A`CG1NvEq}P1pMXEc$=VT)FgNIGJJrl7chU;?@ur5LUrFXCfoE=8{wb0(Tr-ge*0tonn1{-0wqjq-`S{5wn*66W<xfQ<)~iE0hqf<V<Ep>G*&OnK9jRFqw>?>W(8K0>(FZoa$b86buRT@@{5`gii)z<Js`oh*o!!1o4p9Za*d-rS+$9nC_W2Yj6{f6!CEss-2V=CCN2LZ{QE}AEd()4O{KhGzbwIvmI28%rPl?kE}X-eNcm?c_qZrK<j3Lro(}BBNzmD%*;@k58QN{v*h-8oFuD^It->eo<SCqYodd|83prXHk@L>qi`VG24PMQz?La;{PQ|S$h(MiK2gfP4XK+>)Ri8bo|`k4ehH8Wby?a>MK7%gs}-b`nRM-`tx`*8vYTM>VkT4C2@);*_2{-HM}udA2ko?%vWm5;^C7hDR#oo6S#Xtdz&x)aemT@KQjTj~BQ^2_K@g?pX2cAcvLG4J-?6o$XBBr50v2$*N~&}*YHV^l3qzTAv3!bzpgG*8y-&M=Cu5!w1rE?}bW`*in(8497R=2Ssx6%q{}hHC@Sc*_SDI=YPFwC+EsI$x71g1)!IB^*rx>I~YRWJI{fO3@WI6$MXb2uq#dzHlBmh#e{jSx%*q$p|GPjHgtc6g&^25wDZ<<XBrGizM?yeO@{S~m$UI<yNk`YXpfT!t_RHGD=NJ*cHOgJkM3cuA*xP!eh3(T}XVX#20tjc(5Z+co!$Pc)Pz@7&-XXrvHogjea{dWg@&t5FkBE6h4{;@b!OAzcct=TKkVizTAseomky$mp|`UW(oL60@ulpMoS6oZ8YuvKRODK^MlM25oysoVy~89?RuF2ykuAVC?0K>`&n>wi!mrE8i*tu7ois;hzSXNvU{Q#ndYuI)$D-UKp)mtFf=6PYvYr#uOp?jYSki!F@fN*TTbEkjzCIHiY_V5?tPLoAAr?kN*TAmiRDhI(b@363SG>=04HfE3_^W6}M<>B<mp%|44EqRbm}B<U-u>I`Wo-zBx~n6q7Fd<|y133@&Jj-g@MG@5~<2cR%@oAptVc}QqVvLj<H4#t*VFN}J#Gc6KlC`hv&(LD9k7C9fITANe|9i@wmEor2si98X{iWDgby3QX{imp_HVWF6v4<btBOmmru0T?wRyX!C<4j~g~r05bo!8SpbxW)vK<q;eAOL|nX7gV2B)vE=Hk8MLVi;N+n@r>0m?L@&Zm)XaSWDiR?3+te)oUzF0MLCV5Fg=f0y;9x{(UnsPGpRZsBie-%p2F~wn(Gdsf-_Xh`D58!uJSUfWp1;~Yzs|mwoZXo^Cl({BRjVB{B8=N65hD076}Ou5siDP=T-jMA5Za_@lWxd)ESE^h-E!swGtZ-cQ2XI_K}3WP@G*@AejkLjfspRMqR?EQNjkKOY(xaQY(k`!0<BcX0`-o(*`kdI0$0#p5-+i%d@dgj8ls;37DuQulL5xXQv|CTV5|rf^r~=Ae1_1f-~6njC2IsTo_-#KcOzEXNtDvWlhYw-xb>~rqpB=A|O;k0zpDE1d)KJm_&#18C>@05n-HUmo5!dO^c=Ez`|UxE9;e}XHo4SC7VWyv&U8E`7mLX4X*3H^ND0z!qAmj9;U;cD^a1@0sz&{W)(cItT3h4x=K2Mia#W2hh6j-28^PTY`+haYi7DV*xqQHSl(_qd#z<>XS=o6N)i2O@8H$Dj|a}D?GGQe_Yd}7w48Sz?0|)LyUzCh-<{Wc`_D;m8j2VE5;on$9?y{ZPtP*%rmKtXyMg9sSEOh&%2)}D%NR1D^<eMd&5H`n!T##r{_cmp{g*G^zSuvgIB#Ekc=n2Vy1ldaX7AwdOjNsj2m3Et5?rxupuFGyKoj`!&GrZ9{l^dQ-?d)IZY>_t5j-U{iU~D5WG@PPoJ1$h6A;Z~Jf6fuC;&Y0K2aQAVwz0(vt{QM>u8dYbcI2u?0uN<UXYBtL!+$AHlV!BY)7`;%tduNrN@mb-P7pBhi`^KXE@?uf<5fr4vA9HlzSs^=+=l$wA545_%kLT?X4q{n%X9eC>)H2B+Ye0uVQ@vm5d?UvP#{!4qOp8;GY-`J1in{2M4f^Se_W&>=dyX2}cW*uN&dCnGFZVqI3dBLvCEzCwQuZvtW?*N8o!MD3OLtB$E)a1{ohbEj1FE#Wez9D2-lHU^PF7Wsv#~lY1=j1CRo&y&it^yrRu`of$qflRLvmO{~R%v(^ju+UNk?GYoz-79t*u$G!P*l(lo7(w07+OafRmAy>@M{eCbU&0^WZf>D1Kr9u*Ko~6+;@M(agZM;b1VM2ri%7g-tHy0Wx#wif=E{5zF?yCS7S{Vu%I`Bg^kgklMtIl>8y9n^1xis3}PB(gMv3xp)a*$ojMd#?XZbdEIc8|wnvD~t4HWzrnwqlY5`yuljVu{?#AnJxPn2DHf6_>nXS`5!3gy7g_J3;YCkH8ssM#@gc^3OxCCdky{M50k)9Vu;YnB<&Uq{zG)&!KgR0&5V*xV1p1qvYtHXq55s7>Qmvd)WM}ZiD5f443=N5>x6MviW~%!?FlN*|VXc4g12&!48BQFpl~bj(TCA6eIbDr0!mc`9Fxym{)1R-C)yR?6Wvdy@JZ_P5e8Iv9Ofsw^FoucdnGUVh`&Im`Jf}l!duPpOzqM`je9Ni~Z-=lgok&%{$xg-_vV*f7`^Qvekn4@0AMcu!AG<jJq_?_{$Kyc*~C}D##>j*GL8TF|De2G6IJeJ+hKo6#Bz()Jq)NR){6YM$v(HHYA!|K0I2sRU+6))Lwa|iHy0h(l;$_zUsKo$I(wFh9a|ybpwC#9JU&;nw*fBLi9n3gz1aYLadEH<D14>O0H-RxHL{+HZDjfh?~%-qXcd^d0Sa{Q_jzC3q2A2K^hiSj|JF?-psXQxf7<LJ?tG$dz0YPW$Gdu;K8+*vBzY+zA8)zT_utnTG0s#huZIKj7F)^C5XGn@Z6{5k-AV1ujrp6=aBnP{muEAkIW~Ebb6MimL;l+9g6F?c)`TCdQAbMH6Snz*#MP8K&uPGNGZb1Obwk`LX%~I7<bsF7i5f!n#6;ty%=r^`F%rr!hY)og0IbIx6uX`Ni!;rPXJj4z+#jbTq5=hi)H<96!f)#uaMXZ!%R2yB4A;OsEGy`4Q51)Bx2B}66GT&jU6Y|4U*#MnB<n1mp_boaMT-Ox{2fN|M%|uHRsdXw!iMY9l!9{Zxn@SL2q==K0e=ePO0T^R9#xy9u3AEhrYC`|5@7Q#Q@GVJ`G8dnN2VcpFbbGd*6Qj`Lm{o`lEHzFf@mN<KAUgVFDz+eY!;Cp3t{%-aXrX(|+-{_b)yiv<!?hF;KE-HG~mpz=HqVH}BD71$}dA$*(#Nh6-f4pFhVKH33nqh({#BF5V_&CsKQ#9c)8nL4ZDoLv&_Ca8&RI!y5?rsx%=ZmADx(4SxPilHO^1*h{FL)m8CGChSvA!l`m!1p;f#``zcyuEJL`8c#j?jIbSFF;GC5jAj_&$+*L}NCu140L3&VF0z<I79)bzB6{2Zd;9J7-`WT7UehWfh}-}qKU&P8%&d+Re$Wr54*)$`5)aVvEaFH80=2t+PzJ|qGWwA2pg=kglG9-nT?wEnL)s<oj4PgF^ql=yF?RM51h!-<m(`^@&C+|4Oo@XzE@skMA=;@>2arAC(^YFeeEH_%=g*iX5E#MzDbb_Cx@IIGV3wDZHjWFLeu7=Ui!dLda<uS}Xj5`55<+O&{TT^`42-CXmO?z9Ck|~$NSt^r?2p*UB%xJFG6a*=Yo5{1pU=*MD(zq0&!3w#Ta48u(cOj|cLb`>f+HRi5;^ru{C^^;9B1Jf{KiD2uUC_)h~O+h!0*6_fG#tE8GxGR@Q=@*IR=qt$nEs~b+5YAFgV?D+#W3^jm|n_kvR)!nP(`TDUW-~;Wg@(#8^+VfkIb>eMlI&X;c<tI2n${gISo)LxP+GeE<0~CXx2tX&a7)_8c8mYR6PBC6d1O?bMf3>^~R{coDp?k+-BpelT_@UXw;$kji}Ksx#{oWL_{#CcNH%LJR`C2b1#|0^bo8C%sCrNC`VL+CWz6W0<&m%t1y0>0`&*-GV5R>e3pabeF>_CnRBP()R@ZaC+oyp@BBw(#oG7LUcWH9yoDw9#7pQIiyh@IS-vzd(WT0*e67UlSGE{z(JS!Ik31*>iUp~mn7FAk^;i%9I6^43R9UV7758ncS9&YIHiY%fq~%h{_2iK6he2Ewr<`cCf-6Q^9-SXn|W@}J#ldp91$pKz4bsSG^wOTs18-J%9lXmG(}@s&yE`-ziBldcVHoe+lfGcX~-gT<aM?gVHHUP3VoN1mr@?wWh5jip-#tB<X?@!KbT4bCxOcfeOjFk!IsD45$_MASgSsXRgoC(bbN;3d1C0UEW6d$9?6t|6H&hvtE$Mb9AIz+ZzCc$mg0~kq7n6Y1j@gX6oaWa(h`Ml8M(%~H10}p3g29s2oxftrB56%BRi>Jv=<^WJW+xu5dc94eaSoRegvu>c25=O?vrpZi<eZaz6|~%Ad%XNN?@F+Ku+5E-fr*h6Of14jraEdZvCv|_Y%Fa_nbHbM>~aG?O^2Hjkhpi`OLUixqrGvOSvM=HKC~?qDsWE>wIXf&IiO9S=I@SR}?*Z_xf4;(@WY^JR|BL7@Gzkkx2AOm<>}2Yo~~~V;=5yhSP_gt0`0K*-S|~o^U}Gm?v9r1ZrA>xCqlIB7x~9v8=JutY%o|S4fQE8+t-$j%jgAYfGA|({|Z|CJ#zcKQYOxu-pv?2qx9Hu|%U_I*WtR>M3EF#<;|fXy%`>mybp{)#^l#Gge^KC!D#26Cwzba=lcly78Z-2hKa9iuYI3W0h;tL<e9^VF3r|s}KhWST-Tmqe}A7D5-WQlQyYpw1Yy!BR(=32fg+%B6dIstP)s!zB-75$uTX*35G;GWD^VKnAa=e<ro3PiHapsxpKVXCc*R=K(j<lG%`l^1q7Ntjr&-bC}bX9`B<W|!N67GOxbi#P&zalAgN<s{y?!XXzg7lD5u>vttGJ0SGse=5-*3zRGBv-?oCP|U~aN@NHCjCJRMksF?{~~d~f?Dy=3KgR+#ZENh%dQr1`-<aCUh}!)OAU;vGl^6d_=z%Z{vonF8NWWE*CS4@WuC%&(3J5hH$^HIdJsN%n={TTyX#XTe0GiLA9RS%n0Hh9OLzd7n<@F_Wy?s5*O7X~BAl6MdE(Su`S&Yx9)Z_{E5d!BwF{EG>yM{A$AH0*;S#4z4C)@A<GhU0PCinC8^0i^xxZZIR&8(voQMiWRs9vbV{b-rR3&Du@S~PWWy+=`VONNXj16OK&cKQZEN$$u_yCdKk|*C9E)t4nZS+AYxYgc!<yg`;Jm&%$-Qxu|C6+8nsDMwBM7P%(~N?B%riU(@V&kJ>L9@E18b&G5HocDqU-{Z@LTvTJ2f%NKsMAV@a0VR2ZUu{MA|g#(zNRw2Wwg6X^UTo+c8e!fQ52PJc?&D?M63k;-d=6=XVNXJ{fat)(HtbWWhKvQ&Tt#1r?`bK>!P&yhnZ@p?m*WuVp-JGD93wn;loIAQJ8;iMzpS2}P%N4zz$)f*oZyHNRl3CEzYUcie!0=TJWB4V@>#{ZDJsxr&3RP1ZInj|(L5mv7R(Jg1K_Vc=(hy7PWf?_&Tv?j9p+wlu@PK0Q0iIsYSv5`;a{;P=H2@8!ZK`iZS1)em?gY7r;6|^CK*BAn1NG#Bi`kyR`;@>*h{%}C=*KGJAY^lD%ESBbrvgsgWGoX)2Nzk9_=89iAZaCR6OTt@fz1sfpd~g3HeZNdkmcM~4*<jG+&_^iOV$C(%qKJT&1cEq+(XgvrUWk!Orqa(dShq#%-BhzL-$2Jv1pr)D%7`eTTF{>avJpLI0Z1e*WCemys!PvaecXSo)O#RmV{PdLDg5*^^W=devu(}6S#arSy9Lj;$7YM6ns*-P(timo!vBiHa34N&*4OA_!^y*9J=A>ZxU&f^xSQTLa|8%s8bK(Tpfms?q|4a^d?DC$&W3S3j<xvAq7U!#^l5u{?{B*z%C-OY;oYYe$*1(Q_2wOum%m;sFKh5u)d1`|GB{B$+^1WjU#&R4x1<{QRI(m3plCHKC@ui9J8Lo^+>_|S?JSwD()yw~OZ)i1SyR{C&`QZ;>Plmse);Q5W|#zrXASf9z;QFv$QK)Tof@Z1XHhDLMD`tM(cCCBnr*X2V>4vKV_6+E5o+ksb3#`zNLb*nlE9OOuT5dxrd44yU-%ZW*o6J3FG3)KwRD#NiC`sNu|q^trV4A+j9E%}99#^kgH0Xch!k3a!+c-IZ^Ys1K!USL_IdgKW7T2@qUdF{FXwG_2`XIs`HS7X{TG-ScX{<Wbg2DLTmJfT#Zi~L+pPnCtCrq)@qTa3zNXpT@|Tg%@z!~=<v;~w8{gYarJt{|xXgCXrm{PVP<YJYh}zKjRxu(N!;`Vvyf`66^l8?Ct4j>OwAUMHpYRfOYGI#cQ1#`?Qhr^i7v9ug+3)XfKYO;dmR^7SwEf|I1_{8yLTH~_J9cmA@(miduV33YXmIa$8L25Yv6?~7>5V5T!s{C20g{x)NbJg^-q<*fVI_ul={Ou=AHk|;6NlioerM!9qrR5VM*$fH<bNx6?}T1|5qj$ezZ(j_^(=zl?mgG&r4@FynoO@~$y-P@GC)5iXY2}jyScV6i<OEE2L!(G+N!KmF?^%(yG}^!{hPgmk{MbmN8iH5apU|fEt-$qbli&g7XV8q%gh6Bk<4%W$JOe3Lj)dW>%z_q&ZZg8TZcABk`YKok=s{LZv({iaM^^a+9Q-L5e7O4KeQn#PB}kTCQ}Gp#I;P=C26USLQqR_QxSTW6iD5)kjf0cRRfGmH#smnC2XPLXd*m#|KaTc^$@!SfpPValqeU4Aq@0WWpOvwB=y)>+r?T3Z6#4%+ueTiW@r1^>k?|pEnvFLTtEN&{_CyBW$Q0bNZTLY{C(e^+yZGmd)GSH*=jtx1vGh=US7|v+-L9J5XA2w!u3NK)k4|Wq4vb%K^&YRK%uJ)s2rXyhI{tQpLIggW#R-O;v{y7B}+-$L|-xq1WNqtK_7CnM99daMCMaZR=~Pn#G)HDs50ZLxKFf6ErJ53<Cz$GB#TnnOT4P{uHP3UyXqm!y3h$Y+(idESC`&Dhkmphj5rg$#dIh1CFH^cx8m%*-G0s)lwn|kxRNsxG2VPUXrc;<iX~zaCO~3e>x{<T)76xi^_OfdFpQBss*`atH1){vaP6!?a4v$;j3n6FWo@HctzABTK|@nEHP<6qD(p)97EDM;U!QA3E^Ttqf3?<1-Bb(B#5>a1X>9ElDQ`^g2k{JVKL5XuW%YE1Q)RGK6ROZc-F@tA!z)J6A=?5s)z+&wjN=X61-(4gbPf;`qgKrSeZ1wfg%lPkuewAxbo<}K7-1O^3EcJwUJA*6v?NFnX4Qe&QMq$cta2y{%;~S{wHi#0qCrkM?!hId2P34{<d54spE4HtRJ+{y)NXxvMtTlu+MSR{P=eH#xJgZM4C24HKkRMqvu^X{cS{A~n3UD0%HD{;bhdsXN}NW>29+h<hYUb!`t7^t!o(g`oQ;a(*Xh4?kEV-7Yd1z_qU%4dIQ2&!bBh<+Hn=(&WB7fsKCX`Yt5Za%F|opU1Dn$F|3TlD1*fWe0vXcjJeX*EySB=R20=Ue_L~EsrNsd^M^)#=*<^Z^pPTBEevMIr?(L++$fcGV^;OX4d>l(?99Y1H^P>LNXH7x^;;{VRzTI|c2~NX_BvA-Uh(eSU9KiV&J0EC$RjUow+uh@eb5iqwxmtz(x<3EER;lyvI_)fQfD)j=$_BlAN^hd%4`Fr<tKp>PxGTPKu_(2geZ<Z)tey_Ls?I7-Sfl^^io?0yBMwuTYDcrYv|e!@%j`Y!0ChbF77>S@;-j+4A+VZZ#j0qKY^~VX&(~a9ZS4;)UbbKT7j^uQ#^uz!3PvV>>AJ42>)z4Q(h?LvrsFN=r+-AGh<@|xjK@LmESNB`XSRBJ8o?&PJOf<ff$%<|BzAxZi$gsJ1TVsSIGQC4M~N(nK@egkJglAf0c~0dX`n4IYq}v$OcCy>wVt@EDh!5pw~BM7{J|CHP2D+$?}!1h(Z%JTmMlcwosW&~yeXZzBHA8ySJ^+>dQI{H6kyCn7w*1L)uRF=C~o2(b=>Zx8nCdkqWi+KOT`h7o@#?X`_xhcV4DQrAsNWk-ai?P${mj%!i+gP3u76^ENY8Ygf39;v(MB)7tdB!w3jD#lzXSjkS^+*gkjqE{yOX<{7@W0lb987jqy~KvSv&%0GTA01<Z%6L#RJ>`b%=Fs%s%CSkyCWyKEQOrCQInd~3ehJ_=}knoKyWn#6A9ds2s2pTn<u*8aH`aJQP$dmZwfR5Z;n+v1X-pB(X|l86Amzf>FTRh<Lc8i*qX1X%BVz!2wk3v#lBQyEt|*S>6kn@m9*MMhE#0yipLtkhqT7!$dwuW-pxPz%wwgdO0{Lz+Qq@2@(u_(^xx3)<i-)#Io3eUyG{!{<3oo)+GZW$(&;l1zzvPUWT`h1cKn7dszcI<@T0kF6K&1LCMY0Q5-&>^oWo%D#v@PMz+d1Y1A*?8BRPBux1Z>{<#bn{0{JJ`J{{CsOXHr)0fU;^LyOL<gnPF@a=Bm?)+$K6EBGMVMz{kjxNsvRbP?x`#ML77bFR$IJF~*Jq(!^%d%^g0$(d8Yt_J`zeV_7=h}&>NfTo!s1Vfi8Dbxt0dbFEnkw|TP@WPU2f*$PB<>t4zu}3G0w&{kt?e^X?T!Gr4pBN<z?qh6o%ZG`5w=3$=rQ<`IeU5&)J)Qd;j4Df1vI$Zra-)p1sNw3O~uU$B1kcn-eJS_SsLbUVG})ocs43t=8kW)tGvsh%`4==xy1SaVIlQyQ89B<+>_2t>5KH;%6=Ez1+&89U+GoD3OJiUOQ{$@DBjNTA8xRxEeO0!TpOt(Qm(*Z@g>qr7v5r51cJHn&|p3GQGRz_x?L9-uyB-XdjR!pB)mLS+ei(E%=s*#ur%NEMw7A8W+J`U=?Z~ScV|lTd<3~BPFc9?v>9dkdZ?p&{Hdad;e38pM7#R%HI^VKKr)ZTeOCD_FldfbY<4iF83u%u?2!%1ao)$pmnf)u=mXIOK8zqtCf*=8FkC6CFA_m%5j30n#q(c%GO6ZT^wWt6vZG}Fe7FU)&+V~K3z^^h1MiGvy!ef|60U`T7Of`_ri-!;+tD618K>DYlX5bMEa>bQXzc=o`mOKZfq}G1^9*0!4{!KtW9Ul(Q;Ml7}8Yw8sxjawo00U*ySrasu;u8f*F5T9A0bh&Bylhz5jNp?;X!+xBFb1F6_H*0MWeT<ro{=8*|$<ZmG>gW`N$lG=Z-$bI<yHr18@ZIGJ7sTpK|g=6HWq7cNyIIO%6J4pk^u)QcoTC1oAKReSIG-`puiksS9&Z3wh(?Xs`_nbC0Q;nSziBj_K1NIGme1XsVgV0nv_m_>u!TeQU4bXwT^unZlGLzxX<_F~#b&;Id~jy)yubuvzD3B@Q63A91>2`g~`&QQeD36rXbQn-M+Tl*h~z|aY}zlNw~Pg}i2%qE#YRh8bT!lZS)s<rrXY>e4AF!GOx6WAvKlJ-u!&KKOW(etMMD*G+n$nBQtd(>0XmL_AARDdM%?Bl_kZIWl6A?)cDaqZUG&o$DWwRuV@J(?iL$aZnVi(yv<se4VS^2#G(EW*_lhpUKHbC8S$Im+k)Y|mxuo&thsQqs6zap)iR7-bL+l!n0<68BBas^1@WF{zTwny9kpshF>^KtpY}KJM=B{mogfzNv3+a#U9vO5$?%t#|KVd_Zd3a>co~mV37S;iXe?mPgC968!pZ&HiUL8WByj+dT{;Y`3xe$8k55s1mxq7!MKT(#BF{`QL4~qYL^4ceTI%z#*=Yx#L`Tm|GM#Iis&pu6h)@%gMAVrzGUYQSHzthxckn)N#_d?$w+B_(!ztY;K~z;6Ka9%kBddhoC8##No6}8q@R%vG3sk>LB~i9z1pLUwEDw<XUl%&Ze_06R_+x{}KIu`$6{`b(hD+W946W9%G|^{TTbp?!_IupJQD%sVt(6B3=9eQ|~+oXal@xI^O}sHZ08s{t<DQ6!NUFWSv*fRbWNM+0#Fp`Iq)#uoSVX;_M^fi0(Xa8uj9v2Qbg?!vyc@jrkXNjW)Fq@(W@kHfj54GQ(WSNFMH|mgoGIv5H@2H(+8`1k>YhmHC6@7-~>+R^m|{AM0iz|0Y6(4T@6@l-897ixfumVMh&~J8$^?rb7xVcSGXX8Da_hPdP7ruY`M+Utmu=*fwe=X992)km6Wa>UMKGg^J<TPO_9cYm1oRPs6WUN&vOnZvB1#nd{}2AiXReIIOdpS!dC*yMK^bMAoL?YX8@}z5T_Y>P1j*^02at(du5CWe|l2*V=z`lSJAPJ0~9!Wj3{Cfke93j!M`{g<G~=r>Pc>%V9<MkOV>fT25nvj<u0*l+~Rr{ag_Pg0++i#D;Ix62e5>4G6x-#y||ZzfKgSt`hdHu(jl}K!6`!Ja2zW+aWz`gXnOBFt|p`PYji9Ne>q_GNm)E8WL+%52QW_;@zSF8*)dX)y0IlW43u#7PpMvxL}wnMZ|GYI*f2=GTSUYP-dbyzMb~A3d!0-w5U(4i0)z4y^?FyroYY-Iy>o-Jd}?Dg@Fl-j{<{1F_f}~+i%{%yYsU&sjfB(cy)c}$rJO62c+VH@XFx19%P48H%l<Mv-^f!NDQM<6TUZtPaY92c2bPVo#C7uNi>Vtf2k60Y=bpNxooPu6J*!Gel4jrJSF@Io1J8SIUAJEjWJn2oY}_4eAn18CX$eJy;y@`Q8(IX6A_h8)T;2!m)LfA`0@@7`z1}RhK@s^rn5LwCXinox92>ekLg#B2s&zkG1Pd&S}+0I_5^Ip1Z<lLc;wf40<s@VP{S?^5s$UiWWR+=(qoo2gXA#8Wls?sO{GO`j%#AJ3iSI{MR9JdRU8!^%C{b1-4#PTWkIblk!1nBZb{^C)3%OMyPJwHT{@iZNDOPyQu^C_TB8JnAfW*Rfj-4pCTucyl_ryZDiXuNr+4d6zSpYjzQ4Y)Uh_BB>W>>68==3^Z$ffF2|g0mm5+ucIGO4hHNN)1nQu3<HxhG0i6;ZPQ~Ycv|JhFAvz^>$je48zk7iT-W=^((5q>T!&B{G6sgeo_#Dxh_b1w5sFNm|2PZC48p4Fljmr+Xv<LoyVJlW|tHPTl(tx3r;CLq4C?!4aN={%L}RqvW+W(%8{DK3&rvMRP>GtKLoiWW8%NmFUkRzFCGBhD^3T&@G2ECil3wFO;)+;k50(!=164jD;Dhekj<I{cZ&c7)b)bACzGmcO;rOfB>i*_SH(LzdDdtw&K@k=)B;+0lM$-Ey>Xba+bt@xN%6J|M`P?AW~Kq_;w|oQX@U`EpyIZegJH#!(0?-42>7qfi@%P?5ggPDNQG-6jm5<PTci+-;sKy^Ne+!+0!c3P$tbDsgI#JDvpRvryMRRDQwOn&Uj8n<>|SU>rlaWh_RRrW&>Lw&IM&6(=InY#K~-r*YF7GohR|1l%QF@8j<dOoYa4(?AZ}G>gY(`Zd^Z8L9A`%a{+rmZALcd#a}}VHhz|z40pZMq*}5zi}jWJ!+<AR2VChH;G(H#__{1`kMsA_nKUVonxM638$P*xQ?xJAJ!YIJ-?Z2uVh?Aw~R`&&bXp3{ztg!450ejoSdEPUoyl%EENmzs405OUyQn<dV=Z?holo79qxZRI&9RB4p}}uQoNSN%z{i&Ct@rZQQwB<AE|#0BsEOdc$D-47Ng45sx%@ArWP57RxenH*#5kt(d$!74|08t202ecBCHDijr;ls?kHwK%Z!5sIBYTwQCV5lQ{nR5wgt~v{1RDNbVKw$<Zp(W(h2<G+EMfR)y+Oc$(a6!PEmKZS%;gbN(GEsjs+?)=c-vUgQPW*9M;#4EDngU8KYdI%tD2B`|4wJD@1fp#o4G<{YR=01LBaX)p~<2`KcMw077Rvs?-fiLr7T=SH}stZk196Fa`spJK-gY^a{J5nw4?hJEw+UYBp+px?_Kv{dDKLPfv&umwl>|)KT8J$MO<RCsO4!Kw+2AEG~DDz&}s)d&D6(bOTwoI-icGIppYZrOEn*FLs(GU+iSQFiR*q)qY)}kY;%;8}y=tUD_lb(bW39gX<?E7JCcnNhUi>znRItE4>QHBs{M;ef~eql7tReKX=S7(eFl6@R_C^FTkEj?bNsG%}8k&DJrA<>k1;eh<RLnsx#xxf(cI;Olh=%(Xwq%X!FtSqS^JeBQ8yYAI+5VH0v|BPF~1PQ*lmO=GQa*tIvX<T$xXzlU>^SPFm&$vHWI+A)xnm!3&HaEx5rY-(}n2)K(5-OG|Tjn<`O;r%3LxE+`vJW@DK&?M4^9jy|%|Dtyw2pUzN9g4hJdjMH89kqnQ_PSJpWb1pU)s=Ktl6a{&@HBvCU3Tv{t$n#BG)72CR%N`DTVIq>%hyoKxE>guPAsLh_)!d*Zmpb9FCp#=p4x7k9S?ZqhHc53Nozy3}eh)TQxJVK93Zb$T0I4Av079xnj+0;*t8ik}w_Np6YLH85BI;Q0*w86P<8kZ;q*!)|5s?y{7L#Uy26h+mJvuJdgdz_Pgkpy+Cz&HpcGTn8=vZogOR{@Koyfyyp)eF~G?{fI=7&WS;*bC16B&50T%aFMoccfOr1-`!o&O>EjJjn`V^C4nD>NWt!mG2%jTVKo*@#+h57?p*h(V#N><Zeh9l!K!%@fZiB-V0o%vw69JJbL^sMB>j{?!!&xi>-KnU|it*>~J?x_;@<BzVs1>N(#AcjagF5E+R42JuLGWo1QPQAUjO8$BV0wRdQ2b=O?d;L=BGKp9c4?!-KXWB#t_z{$Q02M!z|w|TGQdG;7ed_!8jrk7*oErD)85`f#OLSo1Nr5=R3c|63;<Dtsq=pbekuuz|oD=;ZN;?>)aZ|csNh!Y>X)mOFls~Qna^_Aa#<$HHP<>(K`WxG8Z&+F#4drRDY1#)ja=;J?O(8nYR{wbrr*}X;DxcJS~PSv%>9MtR!-@d5Z7Y!RvwaS7h#Ik?V&-^lPrG~Vu1AnW*h_{gHnZF)woRZW`Vq%qeZMEO1S2^^P|3_EOmwUC!J->3VUb)x!y+W63m3xmW_tq=-9`of!<=)Ty!+MQ9D7w{6XL+T+>~;@pNAP>fZ{PY|w|+N{JnOr7*$ogo&Odzpqpp86)Q^sShlYPkk8gL<=7?yTF_Ed@(5NUzEq4Zdr-S|bg7DSza@uEtKz@DYwcFi`Y0ze&XV`5|$89MA=K`>2xP9^~yqo~%{DP`<upk$`%Q{}W+8tB{jPf=s*Ye>#|ER)_1Ixio<4=@h8C_Q>Nq)TpN>Gk<ICWtcU(_Lfqs<DFZrNiS_>@#Gu4Ma!om@|E)gThCu4@j@rLnwU&hu_^I~VN=&H(;{*wr&xd|_HL8E4~*>TG?DIzZ@_X<#tTWsv9=p?jL;DO2UOS3bGtJ1^Z+4=Lndc&knWpzq&TUemqaW%Uw%0**kPRC}F)Cw~sv7%I;nkoLCX5G<6b(?cOFpqvo0hY;n2GCb6UwxT{p3orI|+eMO2#;D0e-;RU%A*9D^P-(>Pc6Y{YE*8I$_JCpcRPqR$ZDvUrCpUX(M!j~><%+PbWuTj=>B1WqCG#`tRn=G3!DObxG7yJ>8N*UhghP`RBlrTpXx1j#S-a4zo`&{<r9l*!<kHGq&^YJ?lc_py+j$d)&igPrJqt(*9)+P_^B<C^5UFHC$pYv;T7MKa9yfZ8wMUO1bszN~_xc;_^-k?kf1?vVUiW{lJ>KZlh$N)iFDm4E68SGjHpNm9yeE<UoI)n)EJ#nXN~TGcUuSqCr5D(xHX6Z^gJhrT%_|YvJBF{Fr`yndC$43#{LEE7bJehgUG~)&kjNXZU2gc#YWQy^Xs^-DbLYPERxW*!gJ%&Ccs%S++as5{LjU?TFMX}P__c=hS_9141^C)!t>*9GzZt~i+NEFHSexlrdB(q2@=-Pi1My<21Y$*1lJ4^1>xm=-Sd+q&5C-w+s?7!fMgHQvB8E`mWpv+ZdClT?l0CHWi@XLh9m3IR@tum~5p403`%c4`awG_CH8GO3`!%y*UeNPN^(~lE?g@uCxb6$$RVwnvS%qry&~eqZ=jnhP`@10sf?Iw-FbHL!h_hkjzNp~Y3f)X~A@uc$hVMD8>b!f1dx+vZI@BWpqpUSkGm&J6D6cAvu&L#iQ@2zq2a23K{%ZZ61gs3Fn<@pEt0w6(KJ`cIEUSD`4u5!KI2_Y`!L^j$>N<-m@5xT61xE%~@iuh<U*vjml_j1SIbE?NNyW-f$+}bDyoWQ@ReoC(f!ROMc;7#0$|rDHI^o43y?oTPpva7_p-2och?)88Bn}2=0c{vW&Lepj`B<#)o-$tP&l9xUBwh1V(ODMca7~fb3cdRv?T3R-oMLw9jVEaN7w9m(fm;yA4X`y5$R}0ptwFl!a1E>+8MbXa=Y&J~*Hi8M{)dkPw5Y~leQAGtakz$oe&G?*F1||328yX%U*)cU&0Wvc_3up!Hf`N&%7UQoKzeXoYisvh_ksj&ltEybf-jyBD0r|uLO0*obXI*Y1Lvv?j&H;9^Kkq!IKP&`soQYsc{uelIP)?%4I54)52sOtQ`GVphfwiaRanr5D5|t}gxuT|ufH225=v5mUm1czq?JWbL!px<{UDOSpA|$H8q&Z!A@5)y_LQ7K&7PWS)xPT3Q4s;58GV=;dT5RNciV=Y#;7$jz2)R$T7$F%_~#Six$5W27W36Fc_uMTbCR)Z6qpF$xkHA8E}<FEj0D}wd|{-~Nr(_Y;N=Xl?Cl`BDhcQh^_*hWrC5;di^YO|;xZ^tSRZc-*iw!T!y}BT-JXmmUG<JeQ~^N%FIynBkeNF+r>D#mt`5D$T*qhT!e{2ZWr5AiIrQjo&a=z?w<l7%eRPNiM-pU&hN%Cz-PDB1^AUANpt<q%raoZTr`g4}Nn0=eKG_0}z6h6M<Z~j$Y*K=Hv{P|VnE=S)9uxX|o^)VENABuwA@gxdo~^3R2~EROA;a3r+=t^r->2UeipO<x>nm`aMI`~Ep%cVBDro~p6yPR%n_TLo$`GqKX$^hELV3FEJ==|y8^NtnoA^d0k9kc~&`_dk<k%3n=TcEbrLdX9oihPo$R1O1kji^xr?k0gEhJAwmSwN75TpKyqRia=1ovI0Cy5GGLIQs4AmMO0;x`^T8=izm()gc1sF66zrty%ffs$e(X%~{Js(iGT`>)tFz*lK_zv%vOxS8q?|K7ZQ1S(|{KQ-=QR#Bg&3{ZLJz`UsOij(lK<m9W@jM@INbo?9EPJf7%Q*kcpH@ukeiy7I%T4`S}0IW3XQ6j(U&CDj1Me+gjriIy2=v4zYwJ5pa%|za`)+A3L^=4I}^Mnz5z;l@JZN_Z|m*=uR&k;k}KfwY$w-DK?p1e3Duq)0vmMQNoxY@F9f~n9?+NmYDmmWW}9z#{Yg9_8bs(H~C28)ZYw10`|pL8rK>sV6KF<C_1u_V(mVJ2zE6rcD#vqUiA)V9@Zp-IgjbsL(GW;c9`cGMO|??)%QdYYPi+2TcTpuWae62XqGk9R^ls3ujFOz;X<5mY4t&u8H*^b9jKw}b_5nqxvFhFsP6QfCa1=#8QZQUj-Rr9v2b;bls2$b?I|St`GWkXM;gJVXpBuAlOfU0tQ$7JZpd`Zby7evJu{M^C@A@=kUsVJpKcw#1XZ!3N?pyrKbelC&$-Nm7UNn3D?|90*LkDnKw&5DdInhaJ>bpO&n|Gh3*|6rVxHQ9gU%Xhlmn0EJ%oGsGgJ<17-*JT@NoEinAqmi3)iA^@kPRVFXlmJ3)O{`ZNQrCK34Moz-a2$%~dpqcVKIO@+u!iV488(PocgCAPY&NUlLoxU0YUrN3*C@5dr-&u+-fnOw)-@lioEsYiBQ%&r!;fl`ms81{P>531*MR<<KZO4bTYLzrr4vQM7BZ?k!2|&Pz;7N)P>(y#Qzq7J|3dZ(3*fdou(^Ie$YRloq;&3FVHNb~VZA$Vg7i5!+PnatZ*Z!7P%rJu4d}kM@;&i*Svl)FmjvtEsS{HVcBH+U?sg}UvXbf;lT`^CUsKc=fIKGX~I>iYu+z21#BDb<Yxe)JLq_1wj-`muF*mlfz<q!L-d=e>#O2vF!Zek9()raDlGG#Xyn<dJ@!TAZ(fYu<8(56nSlt3IXIPQh;0(5ISh{C$(=kGC+G8WF9Uzt8i!z+^1S4@L*?Z!|R8z}{qOmq6KZTrk-K@JitdPCo}fy*?q*?F?MUhijI4(+t3bphd~$akp+75kOc(LG4KTR8OuRTET*YoEK>WY(<D<0kh2t+)1e(Zb9dGY53Z?}!#fMXAN7H5C#w9_dLNB;86j#yaNS6eV!8a~#WFeT>i@NMy$uWqf+d!p|wT=3|SeO<~VNX9`?9OMh5CZLHrCdW;W`kC+nxHvOW`r;J;TXPSO0pkPa%O$v9favwt1HkB9#ggn36=dFX3slK~9n6iBw{#Er^CNXzXl_Y=_+FVcxzCHC^nFUnfJM|<rZ3UR;Ac-t(x|$v?JkM)p4O|jCBczd7A>md~bS5&qu1Yv~X)SteB3!~?K2|L9cGR`=b<`Ert|6{l2T*A5=6Qyb2;6y1Br?7B_%F=?=j<Ft9MUlNuYU*mO-EkyJ5%~XiuUhJ(TXf<Im-3f=R7AIU-*SbS9qjljO3hvBd`IC^a8!S0Q0=?Z&;OfQuw6GwHU4;>sKIa-k~pQUh|t^aKU?wt7Hw_n8}{BajGP}K)c@E>DEx2+#Gy84-~Ag7|s219G7YD=REOiUh^xy+W4xkkgaA!K+OFhwWMBaZz+z|wuR^gVCYDWg4?E9xUWMK4%7KoZy%{{nJy`p@AnZ!t&ERNb?D=&D*mC8>9{Hp+S-;l;w?^zg1w652UT}JoB5U=E6K^2uXj}0(G_P67c5dF%<Xh4-7s=ge2}y)a@|LrU(2UpNaQ;QxaUixs2o$NfX4@1xF{ynnT<|$@*D|zT=h`JC!p}{hDs~pTwPTHoQtJkn5yg|ViVLUJCFSO>T79Vo%*LwhYnEHNIG^La=33=38-4go!;L&Xg^aGXdCs%>l+oh*Lr4(L-6&^-w*g5>w^{jW>W~@Ho=S>A~Pf2VuywyBt^?mw+I27C`->UTTTdp)oW)Bt01}4;lo-{dG<5Ht+C->)?F<`*8jP_#;4=4P=6eZaMB#35^4}vla?dkL#~^=>9`D(^iTS1q7KJP+a;XRt6Yw_hx2Hrq`?Pi04$%^DW{NE5L~>m21t)-+?^qpyHaP1GK{K<nZb<a%T05sb#68-HLZ$g7RpVlI$bG7(6^axt*p|tRo&{@a~G)N+Eye<<NG$kWL&;Y{Jus!s&=_nTdU2S-)NKQNJL1@uR$^ABU5HH)Y|2vXSEvsn~By4vbK(9H>aY;PMUitrLNplLK-ja6=zaKwj#pMBdPJN7RfFObqyz>UGqC|-n`gWFUSge^O0ZDhsE)S@7Hvp?1VF18L!sxU+eK#oKr-5eH{|qQz_@1MT0g0-53em3F0>DP`Wzh=K-=pv^p?Do!OFtI=T=@6`!i0$IZ>|G^lpg*Gxs7Ot7O~ne?Tnq8$bPEdw68r)c#e^s}A4zK8Njp^xnfcXM$U?_QUPd&n`8Ib0g{M5|{wy}CB&_X~B%sS-zZsuMmF7d>=7Z4)QFx3!7iW@OX&ikqtn*uoGemCQ^y|K~3g6z$c(aPRj_=<mXb()})=#1157)9U-3TZEEglU)=#B{x(!l#t;?C7<0b$1s%<L2B=dJJ9yu?>N6mp^i;1>yV3L5-U4f+jCFYNkUIaICdTa7)j$D2+hu_<EmR&e)PxsQsh{Y3{E8wfTddY4xAjx%LFj)G#6&HCoxPQE_ZuUo$t)3-}HBTh9wTL4-D9J-&E<ww)F9vEqxr_ygc9;>`vR$%S8+2HtRX<QJz#5hD<|hNt+zaAV%n*P~As*pmIH_f18`UK`m$MYGneg@73|EakQMRv6YE(H~-`Y!#P$eESsjHI4%{Yo#hm|mfLq<9_?T>`tD1Fh4Wk>vu8E>ggVKNfeOG19DMyPL*fGwFG&*{B)1yRiTr_qLViJby-SXTO0!sV-20c_Au0gvKYMjlU=jo6vEw+piZj)h_nk(LA0T32OZ<2zP*T)Yi>~<1f>Q)ZG6PGajwS4%VCBl?*^^?CaVfvw@z{=y_)TL<(&};o)hQMuQW~rWhlBp)(tcfB81m7P@^V4RZf1L=Yo#zNS`gwlQ!##MXGO<>y6bC#XfP`uhuTfO1x|ZQjqHjViHtw`6pCaw<J!CA>FXY8ZjNZI?f}aZb~A*ug4;-pmVL8X0Kr+=i@``vUI{3EQ1G(}qQBf_e2Na$7%vZbP#h9DTO1Ras=>LEJpOz7niMrvkw+CtkOFR8@JoCmx-Nem=VOaQ*HsP?PRB=q4=&#WRfV*OZ&~4f8!i87OT17k#7`9IsB*WflNiS-hjLbjDnFofY%X=8RoO$r;{)!2vqC>f4cL%RE>Zi9XXkGGAQNOLK>=_8;ylOoH5F=D>P!XS7-DgfYTV`D_nq~IA_w7V7dKH27ze2K_B=<gibYIjBV{x!ku2gX8-EmU!K7e48h97Qg`8QuUe=>po;rwTh-es94|qixma?VNSALfS$c(>K`$ol{)aJ}|KJgs3bXp`8Qv6244if4^U2#_GlIfE-xl&rfh1MkBT=K9bjXIu1k~tzKGtNrqAU21?PYe)MO*=pqw48W69d^Yn0HT(Ylsn@dOhR&$F^9qn{SV``MkSv?PP)3}u%o$E?jXaWSEvt+yiph!BZjWw?Vu9Ih!4-@5<h`+7RI!q=HuB2vP^O+={4|z((Bp7m@`K)SH<YUh`NypiE%f@|H70CB_Jsu#D!mwl<;*<QcC#c`&>%mdX7|JK!}4-d5GOfzpbsw-5u>>+liE=xy2{cOT5f@v_s$h5!oL`_td72nXz@qo}p|a^i)}{l+6M*5H{#?%B6S8We&fif)#zl67xx&x=@7+g??4}g~N-QSp0?1^ormD4Tl<G6~Q80TB8wh>s$wuW4nV~f^pMS`pb2PrWzg<`qNbVhlGP7S_Y;gWhbN!Wqf%BPc=$HD0?kVdB%ndJLou)ihf+UOece|uA@GMvDQ<^aM<}VB8F$xq<$E6!zzEL^d(C>)7n^G(=v5qr3l4D*5{@<6h{@*Oe&`I<4=jyd9;=@4g{C7LtI89Q;$Ug!FDv?QZZj~xMPC8tXL~r1ggg8Gj(_bO$hyw#gUtYiyC0?T8ZRWlzW4k(R13WrZrt?Uad=w23a_$n>ANQv9V2G&LG>&(wiMAY*hi#6)lb3R<UGaUHuU}1}0S)$(-U;TR|^TEF{9}T_wHZq<2Oim^z(&Be+cCei&2XCCF8iHX?nPxYGh=lkPNTp&ox%l##%fipQ*C&tMBCxc~=x-9M5es?fJDsZ=1{e1eswP2zlOWyKCNx;!>dbHQ26!VveyqvL3#U*z|u<={b2E2+$EO|U&e+G<Bus~2X-u-6MVrtypfCxgkR)hoTRtA{0xn_4$KW}e9=J2>Ph#i6EToE4guMdg=e5u|<l3)HZZpt6(HOoaYF1<$cV#<<YT$qD=7d~;uu142#tHxa}M^;c5`{om!^az7)-Sg{~<FP(R_3(L$iL5@m<OW+?Al8Gw2&_QzxjR4FTRtO}CMyWA`%>oJE_Yk8hHGGoMkz*$qnD{qv0_01BQ9PmbZ6VZZ7c*{JCU(G!Y3Ej5NYQThL{|uMQ)ua{D=WBe<~6!9IN1=r-<t*Ao%4=D?kC_P_KVLBdf%{=j*=1E3$X`rZqYscMJ96%X#2Q$p7`6_P1ec^uX-~{+G@Z=<fh0pK4IFe^(Z_$PEgfBCH5t*NPI69?W9DOB-iSUOjXq>F;#UhhRLuqG9|~>oXeUq)}-~aX7NdmS0Z9%3unG15^Y-41L?5{W2p+UAtJus9ds!%Ed_6yqEU{E;;mM7S5NDV%pADR9K5q!bu&`CN^nejO<I01ns+&Nx!SMbw=chQyID9GaB_qa<~*IwsZBqeF2T98etn~4InU4pP0dqg!$bg^j%%nE#H8uyh~WPZuY@8`jyvaE8v`NM00Ss9PU3bU;;p89T#cy^JfrjDIE01`+~@3_LmB+R0JR?!2JZMJ$9>qr2@o4mj>`xdnblTnW*1mmf*`kkY!y;gO;G?vfibBzHX4slQDUl$MR0JKLU_sYJ<6=#3r6s7zwvg$VdocwWSJF@WJMBIPOSw+J755Nc+YX+rWI=CdU$DVGbOvQ^y{Uay1%7nySX6Wq-@(V4vyl)Ym$yEl8$VWR>+QQKB=vje0K;N8si^?$`_kO%!vgqkG+z$z{#Z1;$Uw@(-%=n_a(z}^pL)aqN8u4XfrDq-ih;<E#`CfXoT&Zn9d5*bFFIqe2km2LE*Y!IsTGOK*;2ns5&H88IMU-YT2zg)DNkNj&l;CAQ~ql(%RkiHP6~IO2gEh^tom9x3;O1in95Mwe7N&O`)PedKqKlHJ|#4w}`-R*idu2@g#^d<$+w+`t+gM_wJ{pllpqiwpsxguRN6@^~u2ORk}f2g%5Bj3RHEIpcBXz*2PI437_`MQ3#Q<A?DuK1Z)VE2HZf`aR0|oR%C`8nXq9FEXV66{!_P<Vx>->BLD5dH6}V#=;9HrWTSQ@8JCy}7q}DqG!vd$2J-yAJl^mMGqOf)XAPXx7nnZq=SoFH%-FGz&sVfQe{h3wv8?B*snPDTmrG6*W*Bt4s#E~;!_5DFui$Zg@$<}>3$jbhvI~gHHNGj&m?zn<;%YCIcT-a2_1+a5ZS&h>%~hDf;V7iVc?*!1i=fIkDG^(9Xm98q%1XH6CHaNsuzE>51pVQ#1HQj5_kWf9bFaAwB!nuCuOZ+Se5;79E`c^gNrR0=uK>gHS~CsSdHeCrOE>mZtws6;5MK8AM&X4GFGHdwLkH9Os-d%_=%(G)F`NsrQAPD_Y7iB6`u!t=0=`d306c6b^7uOqJzX<}Ud`ufb<DA{=UCVC>aY0U{P#s8=^SqGCrU@?9^upU;>y1G)xMbD<W|zJvV+iuDmSk3H-63EusY4=0^Y?Ix(2P~nMg$f{3SAA#cZ0{wyhF&sP2TCPN?J*H=6c4C<#R<O{NGM&%EYk-ZO;{;niW<`Gr{(tT#wtk=aJjwwhPp(>LqsO;qyh@bI}o{gY8)w`U}Qyx5#&JgijQ#CgCL#W%CQ<~0`p^6v<U#j#RgLU5Ymx|}RR|G(n@|H}WL)Bk<G@AG}1@B2av|85hX8(9F9r;{D{>Qc>kP&bb;?Li}byojKfd{8?RB{NYe69qDFLnK)k^%5X9lv~hRng%=LBP`MJ%*BRf*OP_HyLpYRbi8rbj+%VSPjC6^Rz1B{SGOAJtwvg;LMOIhd%NGN;y41ha{VKfoXIsV(9$CRBm-bSXwTE~klb^nyA{>Pit1QJ^+uy5Z>HhRo1{*Jt6%9uYkSS~oB_)kllMz}$r_?o%t=SrTBOW-e}GlWt}F(v0k-fu&MqVVP^uHM!(LS#Td#2Tf~$*=v20J;MP#|wnne(7uA!Gyw9r(pbg)n=941CjPEsxThA@%2E{S5*M46Z5xGTCkFOn4sr$k4G1Pj%XthJGF#yrw0?3=#alnkSy_I9gK2V!`kT)ohN57}=y7m9m?$N;R}>C{S#DzmOz2}s%4g&Yo$6UFY3L9%_=4UcsGXLG^(*}SWIdEQl0!<^+Z2n!A8a|!FExrFuVRR+Y3iGfKjDKPFRybT7|B?sOnJJIv*v@z!qyt4qo*6ISubB9^U8w{GIFtd};xYH!qUAw6DbJ%Wh{u@k`@EaGOUddD60;m@2Ihz6doDA3JY&iby_nWsO$s1DIlJb<ezSsOFASNfSUpleK0lP&4n3ZXEhi^8%;u~8-QE6raQ$sh!Y*~_+8a$_ld8PH<jA<J7JY#K!iB7$~vch)b)Rdfr=i==LBGYGiZq#mz51AeDZpp@a<6bG&`31E~PWqM|ww4-VPe$o|mt!QvH<0lzT6+h(Ay4{E{CGmYM%X=0PD*(vpMap8jEJ#t>6=wFrlQ#)+biEBf@REOePRJ#$&jw)c}?wyCxEi*1lNgZg#o}7YQ_Nnrx`|S6*oZ#`7xdHKBOJ4Ia)hqA#XuPPuY8ET(w*3n8?zyf`&x2Ia;PX6N|upznu~{PU*Yi$jF!8kU%fXDPe^=ZK`-Lm7N!jv*9s6LZ|+tLZ@sGtwi#3gusvRrvKgHO}CFTxV2Nn_K{|DP`)n8pgjJA3(kGi5g_@7is2uweBj+yg=F4q@#`OP1vkeP6g;OF2pvFb+rP@^FLRf3?>j=Bi^$t@=P&km-!7Z)b}sL7zjW&AzScFZecfr3Kittj+|mC|?&v?kuljxcsRmNx22g81T(du1vp-z3KU}l_j;>j>D{=Lxhq&p;wV!%w<wjj@G{kkg_QQAk!*}~1>AStrv1T0`v!)J>fv|C!dTsGj#0LFqjeccix&ICB+=AbBv6g=~&+Q^j-!x~d{Kce?#hQNxfOn@XkZfwl50CK=kMTEpjMu)W$9S#OW4!jmV|>FTJo=6v<44zej352*82{&bj7xmPAO8Rcaj7TxqNYuhwi{S`fgffJhIjG+{~`Y0TY7*0Wo}eabiYseSh$v8lix%;I7#5qx)|%#`D3(OT{^zpOY6c`XhKL`(FL}@XGcs;SUD6EKDxAit$Asfq&+poYC$<fD5hgoXD>Yp%8{yY$a1IQoUSI*U?ASW#yZN%xiUzpfJ;ERMV)torN`Nx<)v@YRqEQR_#bfyv1f)MIi$5}wXPi_%4Y}1qf~;pOpj{Ng^jp(7$yJU<WpT=PSr;xS&#T*AIURZz;>htC+#(=)wM$9+Wpq9a_rHs=GWCkqG2v7bI4vA`7`Ut-_ztaMckBwJPM}TQB`n#N3)K0B7PL_^2e-qxxD5eK>lKt9*k7!!H7*==5}A*?t30@IC{r{aS<)A(11L>sh_e}rKDov;zD$aVf~BC75>{6Ec`=D7N(`fz9*MT<k(zh0}-<URnZzJ=)&oqmcJ&vg}Mq-?<xvVYJ*RvYG&P}SnZ}X2TFF}Dee6+pQ{ovPl$1}N-|E_2*f2}z|ubM1sqEtFw>J(<mfCNDxg>q#}F7^n2#UG`EV5@RRy@e8I|+F8R>yEi9go@ZxhTNI5Ks&D-QAGOHb}%rf;m44SqsU>7}IGPUfTsr$>q>L9Z9<psUxbXTixB1vN-j-f@!o@Yi2Q#JM)O1p0vZ_9St9YzhGk57u!drV6e{JOUu01h0u9yRo}V@&@V|o0qsY3Q8Gg?z(=hzNS7z?UXt{_pyi3r5yJ1`<+Ic8>dyBchnhMS4n4_J$~hm<8fVrS9qm#jBZl?W}|+~%EL4?C4eH#EyK8&a;g$(!wig1vqlg}F}%jPT~oA3&xX^5ZOP*{lg6!@Ot)6vc%R;?6YT6<)CD{&>40;uX7+?=+1<~w{gSR696SZDn02Slr<WNaz`Fm$e=2mMBK2-86MYt_^L^a=LAUE2>J#yf4%d!S0=&=0RO`ymT-7sIjZ{i-K;-4yXu6p!C^Mae(@8Z7tDQz2LJGB5%k4r6!7%%(h%F1h)wA`rXyephuaZ()h0%il=h)wWxmUy9EW6&l#_tv9%e^)F^@#pkr~e+)oqHP<)x7S(@7Z3^w5{})mEH7^k9Dzc`1u?4{EfzumqNoM+989Qr+@k8SKa(-c$xNTw`D}9+vY<0sQ;i3i$47II}r%|mnV-eppSb$<1K#G2g&&LKY?bf8-5;!pNHY+VfcSC-8FvDoke%`wQr(3;YvBG`@&>*&&S(7UiTj{Ew2B6LW|aodLBkS52K!kQU8<adF=;1TlCzhe`k7DOf*l;UznWl`S`-eC;rc%@p|n)h{mlOjXaD-9!4V%qwy!NlSe<+iM3AFYJb8yQA|I-O1|(ixmRcC_@>_AwX^nP?c8bYSU1-4FxK)g*77jc{^a$y{$u^w>u=*<wEom4m0x{dc=_F{<C{9Zsk6#i-}td6|MWF!-FTFT@hA`DQ69#lKY3k0{;{sDb^U1l-)LQ{O*Ox&zwolYSI0LfyM3?0wt`0e$6EjYa;;l8*7Gpd^Dx%)FxLNMezEa`Us(KNeeK_zUnn6U&n><%ySUfDHw}E#U>i?k{Ra>EkL4lOjmLQykMl4d=V3hl!}t(8o8kVJFdgNrmJ@4DI1new^KykFd5ieM`3%Pi{|aBaivK|{$Se*<&Z*Q1@JbxcTPhZZABkI?!y?qF);iWlIgp)bysDhXPInxKbaxsLFP%|vHJ(kK!-&}2Dg7tMZV?XGot-1ceY5jUd7gvOI2s_DsztZ>a7($f@0GdjZWo+`Fpb$XY|nKK3{&`GVZvl(L26*~^@o@_)<J0222jiI0M;{q(eB{zA;#tmv`dMJ+H?+2I}WX(^)-!N;Pfcx)puNu;Bva47`XQ%9E#C({z>%U+;<)$(HJxpP8>N#O1<875DSs0Hcuri47cfPJ{Ef1@6#57<I$7pH3u-3)!w*7ERzXdxm2--=J7T-6yrc0f+}5-18a$*q_>w&+&MbzBbq*prYPHoEe*K2LhQ$MJX6)wiJrNJGW(9<_#YBMXBy<wt2pdlN;CJzqn<tpTQ}eYeFSP30C6~qoIK63C=!~Y*OWx1xhUrn&XUk8i~mn^M|JXOSEsdfyJZz0fW0o~B9<id&TDtbE7ggmNWAKDUTH(8m1ff%GcEXcX+fGmnDX7)LAC1I*H)`8j*H{x7Zj&PQ~p;swP1kR2CvN-z9IEY9~Q4pqi0?}R>!)%`FOA?cRQd=x1)}lKPC!RN8;en`c=PPbKGRsQRl(SdBVe}JDT-EoJfvn$`HrLTaB7C))Pcu*8ll)-5G$IQ?SH&M#Ixk_oOM|r~~IWXRxy2Oyl4p9FbJYyB9inSznN9`ub-ixYaJ#pKsLg-%J+1eB+|&3U|O;@cjgoA{ntNFV;SGJ&|<5MDhh|m$kJTAG}Tw5VOO9<r0dt;YqYkt+u<1|7Lub1VrkfC;wlREj#hA?}9H~@c+OrBm-4w!ZbU9LkEXQUivk^Hj`Z8#V<ALjr!~w@P7VF5|i;?2K?fe40xT)y!-ZLt&q!H^D-}N6yMm$z2X<%@N+PVZ*2I6mKLOSR-ALXKBwz*x;{^nyO84P-k%dqdDrK8suz9pVisWf;`U)eTbW9#M9tbQg_;^|&e?~)dI+@miy+if2=xMlh6Q0kA2^xQ^uu@0^Zk$)Okd~?|2jzZ6jHquQp18IH5Md06l1V2HZoU!=8>Pt1WBXkzLEQ;i$D}^WU^&rEH69E^ug-=pRuhaQ{LcGjC^|e1oG?VCd>hl0xN|nuN3CFQkc|AVJ0iopT*7imC{+Qlul)(blxhZlVvxX;#-FhTIn%_m8V-sF)`>=QGQSSr>=FF8|}CkLj)1Cq=#7BjEp5`U}&L=#k{lN3TI8lA*zy1!{{y<!y77Nn8m(-Zt)OF9;Tt4GS&Yxd&e)_sTb}vj+*&o7(LS3(*-nmyRKz=mFZBXFPUzD#Kb@$Nm-!Yc92Nspv$oDi&7y|5cEjgU8z-T{WMbi290ymsoU8ii27zZ=n|Tmq@S=Nnc(C>k}E{3?2bts=jw(qL=C$EPC|P7c02iYeeYkW?0u`I_mwBuf3uXVgyf7UzTp+ltl20w>3^GGu7u^1hE;0Uw<$?~lp<7umOn=`GH5BO@KY)Sww^tAyJ>-&nyPTa@~_)XtM8-&;cdU*jKQ06VF`>m$i9N?m-po^UzomV^yH5Y;_aO?{JU1^p7Q!!3De609B}=g<ap7Se(p=F$Jc);lckCU)iTwBYIeDxTA^OhVstBLeg>LvL-UK(hpgH`<=l~rWFZZ3fXZ0!+q(x^=;5j@TqSFIqo}2<EcBO9#iG{4k`1hCD6jy5X$HRhGBS=-Uk&`5D=8dHTgSSo;{|!MNu85CAD~xuNrKzdd&#H*^=nJM(3p;uOf8BCUWK;PRF8b?k+0Mo{j6?1t7kuJSdSX|vjyw3pp@LuWxuq`IE7ap6}y~1D|9)1RP3^!KaWf5#autZkrLb^2<{{=V3Kob`8IRzT->G@5lS(rLKP{UZ!mS(sIg1r)D9ykcYzYQ^e^B1s+(UBojVt>1knGluAL{l1z)w`n-+Z2f^S;z7qpN;>!Pmvrt7}xx^KGf8^FKv|L0%Pb>DQ|H(mEl*UMYTQ-m#Hv!GhY;8m%#J9{s0DLzWt;5Fs4^PI3V>#K^@yE@l5`6DV--$oGRBgAhwLA<2AH)?d`*Fp9*S_w4H)GjZn%KZ*+BdRdZqta@+T=b>@J^R*ESJnJMLbhG>#~Oc{wEen85MMnfY58?a^c6YVkZ})A0u_Xj+)V3p6EaLb@{A85-z`?rHf?;|X_aMHb7Mr?GZbzJ5>kKovw*&EljC6@lSEL0p6Eus;l;2QT1CQVs;UDY=6y2mw8wqxFz?w+l~q4cf01Ktm+d`4sfyW*22TBJI-CJcm#cX2K<=IJg!I{;cX)D+lZw(7Ib&|yYB8<4FXfCk9jQJe#fkgqB0<!7&e;=RGY248WiM{7@%r(G8n1=I8o%d`K_4b<NsZ5pLuGK%N=FR|FrrT{-|p?B00NinbzNPxcEx$jVpziiA(%HaxMCHzeQ7t6X7J<hW$D@L*Azu6)GLmrTrd`&m>v~hf>OQ3?>;cab_;8ag)vkxx#X%@E^Cu&zbYOE4dr}GAk(|FoF1lL5H2Cc_;pXbtkWUmM(i1%uXgU1Iv+RFV@)S;9KeczQl&d=EC847kA~gpYCjIc)Bv01P?zlBx|u;)B}+0jrHMGDsx*xoKUTI#I#?k5#e9r0YJXJ-E?;Dder4*yLx6~-L`~s^+!Q5oH&^EdYe<SRST&zAg?-!$Vm~o6Zd3c|G@)Owp*mCV9$3|vRb@5}@zRC{I7IwmG?k(`@sXl_fY!Bxu7TX0NW2g2;5RaIOx6l~%&G}v1dX9<)i*@brN?kJ!|5E0(^>pf(D~wS#knXJmC2Q)y5kI|bNkRxe1(hiqVjxPTnG5)-(ZseusL2l#jMrEwpNFC7f+1hE|aXi_jg?>AN=?<jH7V0iRoDt8At^!aCL@WFP4KSl;w_g&!`(Rm*Q9!H&E4I#?k0XY?Cnmo~nogFBOprd@=4GZX9KU2`+odw0QF(7@7J)%-th%$=d-D{3N>@d?WK09_(hhzo+z!94%&g=Sbi1{Zkemf8jDtG(j%zt@1sRqz-f`DXfG|{`gv5yu|2SQo>5DHqa94Pal-j_RFx)A_vX!Pqk{XLFZSQUO+sJr$={OL|F>wg<}J{`{Kz0WHMq{g{6`iz=bE%_uH1^&%oqP7YH><;Fwr6RwioPdF#L9q}zSR?aRn3Fb6gEib1I1EJTDnTEgBb7tb`_<8uD+-4Bkzp!y;M;}{&~K&)F##>sGslWkO?7hOk2*K*NLjhf*A=OzNkj<QlwN8-5%WBoP>hqDg%K*V}?R*`$RclDqIm8?5cQpyx-l|q^oE+chnDx|tD$^o9@qyhTa%au#*^G0mHQE^hqj{7PK0bQEOfH3^G_a9zBpcbBor-nFPgatrWS38m1c4?&YfPsqqk5aECJaPu+g)^20@v!%Nc&3yzg+=DN7-nSgJ672uRF<VNYV<~g--zEV%C}bS(aIeqa&-M>nS9!1k(^!@3R4~}IJs6+zZS_h(4!_ODFlE-yzFFKMa7}8=h<MbdS7Oa3hl-n=+ZM&O;e7fh&w*l*yP%e`9l(FIj0j_VC-8raR}>B7X6tV&GI!RnZHPvhDX9VQ2<wZl)3YJo=GJNQi4_TnomADO>!EjMYy?B;SzOJ3w`OOge<qgug<0Dp>@zkrp~dUyf0?C5XqUv5#?4#7M@*slOg8WV0pFGh5g1#`_`TtP!X@%uj#Pa%O{rez9<QRgEId?jV?$q=fa1~tLcd*%&c&piG3i-IiwdbHgqj}WDBj3Z{9EDm@!>M@KT*;hOj!R#AV`L%L;Sr?LgJvrp2$~L$N4xu_AXrk!7JkYj%4~oAO|lYWo(NEW#q?osXy@4wkQ=3<HqAnTwdl%(g1ZO%9K^upDu2OW=ycmB@*Cgqsdw?X+2dTrWVbe<#R|0_4Vbg1lCMy!M?SKPo_e^qnBD7a*^HC&-TrkRN|L<X+ehW}|5V^v1V>zNVD*oz}oD7r<=^|CUsLTWY^0so$2wZ%N;`rR!S~^=*mymXv&33ce-z-kNN0Nwe3|>Y~j?p8^p?Jyjo9HXnT+M0y;;<$2Sx*y>|5(*nA1u<&7n6-nJ0s<N1>$}X3;97kbsFF%Du&Z39CfwQ0zX(6Scw}eV#i54H~U^aeiRatdCBf~7vi6T~tBDCSGq8GPlNekFf{!~0j^6ETCsELf28A<Ubd4og)sREmk-R-LQx`Zxj<A+CFMHlA~ghyVkOuk)X{4RcNTL_P)URGx0H&`#paW)zjiGLHQKdcc|a)&QeSbtDS3S41dXco@eYhAmy1ML?<`v!E*9c>mtdDZn2_fWx%P*$zHk_yS`mhi_TUY{jUW50HCj_{Rc9(_Bvm>D@fAvZOESi8mi?A&60@(uX90Y7&n!Ff%^kzzEpqbA?+%^hFjhpc1Q(l8x*UNYkZGij5cKI?|<0JC;YUlWAP4@%B9&`KlQ{$IDUrUT{fwHYrQ#j{8prYDhhrgozAWB{$))U<N*J<L-XLKPzwl%&$S_P3}<9*q|TFtnrL_laRhK|Vn=1y>{2NBGEh_+};t25_%cz%X-biROlPd39SIN3QH~wBnf228ol5#}WSGUAUa50W_-clDIBXnB@Fv2BeI5Y*ayvaQ?t^oOFG)s2sZ*y|7%GiQVnif!}_&@BH?gass!~O2BC>x*QeFHLjg`7>6>j9`YW=uOW4)Neu%$ucG%-0*6t8tW_-ZcpfFDQnkGfv;kvx`=E8OeX#dT&RE5P3{S<W&RK}M)HaTSSKWd{Kw$G_^sNIa3bog*GZic&n-vQ2d#=)aOPx^Oq9us)S!tu~T}5r2%I@0tE2&LS%PmqAVZ!K<3B>u!0TM9N+TJuQ3<zi2TfQ9OobgM}9_mXdMc?&tdXiL7@T}JIM{wh8i`j@iY7g89M@N~E5UZ2K`Cjr_DU070OH1~PZwaDlRk38+lWA<uIcKL<md+}g#5z}LQF5lBWGsJcur1D})T#$eMkF2{hohb|n>15WFiUJ0skUSRw)rVHp|evHozAMDauTc9GcpIJ>$2=g7++xKj#aD+3z#9z0DO0HCcH&J>ZB1+_>1iiZ~net(Ffami_QpO_Z13(m!1^xRlIk>Nz?kEY3r!6CmAM}I}Cl86Dv*l(~yY1Rh>arKO7#V$5iXBDZB0>tZWh$BjuaRROMHUk_Y7+UhvUHe2h4bB<V^q{Akx=2!k<Zj<6q(J43}AHq!}oh|?rkXFN~AS?HXNdm(~2x??13Yw_Vz>pNAlCmLN%CSi9NjJV{0&G~5qq@SJHa0C-LHFrRrBEd!JQT%Zt%L4l^&sr~-1{M7VO$GHG8XXS@x)IpmvC@u)u|6HoA?1i5T_7j`uhJ5kz-2>i@+wk9mk_m8tNRBlI7D473qjq%+igC)#NMQW(J0qd)tep?&UI&qp=lt>!^F*M^@7(|G{)Sl&WW)(>QD(M!ECCc3h<Wmf*Xy;6FFnRaDz{+oj0%5<a7Q`cHsIT6iIexf0y0mfBeLohVD(Wso*`Ozxy9Zy$yPj#zS9nyBe;Vz*!W=73W=d>eSaYnc>T=I!VY3YO4VT&>s$*<5`6BYGo^E)3zPPQ}-`IZP<C$&z(1B8|JP&QGr<mwc;$N_s?cDdhomOvKtc4^-Ifk9;w{?unTYAjb-MO8{`EMh|ZOKv@L&8o(F$BwMyAu;N5(f3^#5u<n}zdp$iwPO`O<N(q`@qnM4H<l7#t3M5Cs)5=e2_BJBY_Ju%10-anoM=d}0GQ<u*OBC0}`xwH3n`?(bod8+a+Y%f=$V=q5Q?B$2-$u>L>C%Z~2+08%s2-YjxOYRY$^2LXJndpx-eUR|}vsXvPgWR3P&<j>2YC$BC(IC+ds_Xu-7pcg-JI2Rsj0-wyR39o;^D@yWqlj})YBL-?$;l*~0AFaxMfQGroD*@PV4f!bdl^Twz2$w>dtTnbc#-Z}`9~+|(Pf$A$<kB1cqC7(CCsOT(q2w8vg%+$@J4A=N!)>V=z?Bx0UgusiBL%zM^qcJe{-ajRO?iBqX)|DwA-WJz-8+Ty{BBjv@I)yroJV4aZ&8QUR%lWIG&Q!HmL&9bB|BhvrZpXlk<JR+D9E*Cz+l&f7xPsv=Jxdc~Xd!$+-|^P2V;w_rncgxv<DU*1{~7IRj~x^xBuC$PW_hkl7P$0u^)UWi2<xthZ2|!&f9f<35zby43U3e`wBzEqV~;i>wpzs~vN80tp-ZZkgSe$$C_|g@}Wd%)U&?6XUyAC$#f{v6=t<UXde>7e3EJYTN*7?OIUR#s}!^OW1&)z2fxC(q#EHXMTrVNQEJSh@dL399d=Ak0r@?*y=bp6hU5TPM%81Lq^`F++UabzsmjjHFYl}C-JpQz>$Gy5)doc4JB`J(JP`+p6WbK%S=}2I0vG9O~*NG#dWO|#~a3?QUo&@f_8%7wfTVsnDe<Jr#S?RIkGh8$l{sH$?N4OxvPWR)o$E9!l&uQm3{H6eKEf&H(S5TwF+#ga^ot0<JbI+YmSS*{bBL9OC7xVpm>u?g%-D1gQWF~iRWuyuIuJTU2Zh!2G)5clep(qvdMbsbv^sIVIJRjUc7a>`z-?xzvjI5hpc977Bw5A43P*-3L-QoNM6u^wThL-6*ITdJ~W80xkJ@a2T7upYo=L!UR*)-lv`N^3{fGe;;FVk+K9Wywt{H+ociLD<8J46z)nX8(0te_Y=5olJc&aZLZ_46040Xz!;Wl!C{N*8Wj0j5NK8%9z#Wk819PSl2Me(`9eb~i9`AIH4l&rHLojSEp2&yMo`zC#VojC5<jqHN|F{gL{B0b4{Smv-HuEjQ23!)W>L*&-hX4y<N05c2Q~kJplyxPYr$^G$uI8mGqKGu<;|lA8X&V|`#VCQ+Y38m`bV)E4k$K3T35dr=Y9{8*(<*0R`qJ4x&nzZRQU06FRLc+}0Lt8{s*k)5YSrrNRln>&-o?_ox)*)a7H71f?0;!fo2lJ^2q+FKtd4iW<Dv3SD>Dmy#HTtIB0%}TV_PR^ULy+5NTMqt{Suu8Z)UX=(I4W@BHf|@Ymm>cEF@Sep^AjRvm8YiWv3gXujj%9=b*Ar;DYl4O{><X*T_z#i_cnlh6<B27_>>puU<BBaU~a5eDSMX{K^+|xtK2!)vt-EtvFpP{I(c6yWl+dq9+|1UDjo8zat7=zLE{=d2;$9QRUn(e?=ccdDHY<S1`b`rJSmA0n=%wT)@K9?bpjsw|9>r#bH)9d8TLt`qo@-{pzuI)+=j*&#2FyZNE;ljkE1b558Yl2j8c88Mi+5e&Lz--{Hjj?{eO~(Ma!p*8Sa2y8nuE?!Wnz``@oQy`EhN{~DWT-pW{}v&;5Md(5`C(BTqbFz1T{d5p&;QMMQk=dhq5^*{sTuy|^66;0k6W*&<gYqGUngSOk?MVneJkC(}i`qXOw*So!a9r&f(apDnARQ%U4(VKD{M0`wf9CEPoRkod=W@I5Lv|x!wjF>G;VCke!Rnf4VP!(Xe1_`8sR0p>@HSY%7Y}OQ&W3uP`s_~39opU9Xknlx|16TWCzUM_TOr$S#2^5)1TzA>e(|7l|E|8LkOa?iLMjkQYeHy5CVY0Zy2;;#MZ{BZaOsy($h`WBQ`u@d=`ZC@LYUD3gg9b-3@T?3f0vYQq=NyFR48~ltl~fG53xZ&cgtI3_wXk)g)K4{>76b4x^-v(v^?Ex?nsbwMT%4brK%u-N%qt)cZ<mUA><`E+W~j8xDN6EW7J3k_HEHK%#kj*K+o#6vyJ%zm5^XG_jCud@0%eIkIcK&mszRYrg-{Pzv}_a)aDqWFjEtI|oNaRY6|sPdsy<ksP9akjZm#(0?Ujwvw==g_eD7Q3VLSM<D9VZaD}kfk@W_&azf*;)MI!4pvfVZE?Bnky&lV6d%}K3~VZZx-?Y(Pz6G^fz`kh}<%k$7v1+tK2`GSm@Z3A|*8)y!u7p&v)($=M9%a$dfk}SCy&Tl{QMnq<1R+VJa(6eXtbU!m#Rh9S5xMo~t<J{XD-$GHDNY;}%F2hA^B)bZ;Ji_*=(0Mfl0BgoPE*wur?YQ|VJbwA=m*(@A{~4Z5iJ>b4z?;o@G46%yXH#}Oo{gg@T4~uW{Lda!Q1;I@f_6&OB^Jv*%xrK!(+D~UjX7nXcail~t8_7H<0fA{Q4{x`rr2_J$qi2avM9eHj4yi@VG`nfq&8m{1N$~)9)X?C9fX@Xq!)a3)!W67C-iH8lX`KH>DA+77%?tFP?C^pt)n}Y;lb0j)xHQxa_o2UY11;he$#~p!WU?WI2^z5m*tz8^(-j{;e!ka=M+{;1ma?G*PO|k2(x;FcrxVdBu3y}S_s0@Y4p#)BTJFR&UZu>tHs})RpReXPW=4}0eB^v{0>6!cM^lW!~6B3@NW@@f0H=;9R%WBEVyDEm=Ufr+V^A%GviUuB4dMr1IGdxrl95~fyG2~6PC00Ie2+g3YM5{7l`C)36zWn#6XJKt#6TiB!4<;)}zKiKd@!s=dUrFJ<3W+pFI25Lr!*eq{guvDC&^#A|i|(rO_&SFdRgviqnu2^zc7)BG;b=>GNmHj)^4<TLkv{9kx2eN@H<t%A)>I7eS2P;!m^wb!!l_m5jzGV-ZJ83560u@+FyP4s*0*BvHXz(>_9k1^hb4s`Y@F5tHU`>Z`+c*Ux@Fet!T#A${W8RlmfsS0eH(r-Ms1=uFom>LDD+WoqE*tb4jh?;&=kb%q=kIekWS5nk+2-L$>0;Sr)2MQIsxxigw_RuA5zqhJvX=&&=QPdlDNs)w1a)80|0w_S!!D)|I_0oRbZIL#caY{oQkpiLb3adkd8;Z6|l*grpJZ4pWrfmXcL<zTZkq(*kR=KGmNqLR8w^Ml<b+wcJp^C3J+Es}?9d=G>~-<~>r5yT6CX)qpP*lL!$XaM#`lW+)o_z&R%Y|w|J2<KWuee-cAT09%HnAo;Q)AL{m^R4!127|ps7(2H6O(a743BmdBI4W>BV-~}KaRqPS4gLAy$edA$FvEjR*L%UsjbC0Qg$1PWBNH5vIxpZ_uBB_H?@+H#el^|m#2*AMrPhY7q4qnbJ-ntZv#lD70Jhl|>_=6_C|@5<Sm!Lul{fS67{s+(VAbCd(~G0jh~HRTTl43m?_~c>4K;#y$LkVD09!4z2F@)kQu53oD8jF1{U@=5{$<G96H=*CP}|&ExAF9&hLsACoRP#CE~8j6KsLmf*c-5r?B>bA8>xlhr*d%2exG|9yPw{xy<my^<guYDWCF>t@QKl=#_?;ATtw9*KhVh;u5{gt-Fh$|O}fcxnoAB7w)x}`z=CSC{ihf}yG@*(BKR)}#i@TP5d4BLIuS~77@>LMoe19;P)Q8kk;rQdL5L*tg-jI&kh0(hoO~|e<Z~U<F>=q*;^orMMb_pTe_$33O;tJg;4Db1)0qKk5gx)>95yYdj?-j&0g_<8xkNlt*k@)l@0m19ZjL>uQxWq^BF7|AYCdkpV>y&i(<wa)&Nv7$sNWeljz^F@#X|hUXmZIO8Vv018zutlvgfcp<kO&f<ItWzglS}|+nM1icRG)eu3^pw>PZ@M&9kd#)nki06{lFEjv?gIjTfo=3L9Tw(^`|8$oep=y9oV5yVNH9G?7{3o0)MrW72(e{N`h@ahv_^LbUwQh0@;pfnBKys8k6Yc9%CGtL3y3gbY2qtZC|IUDjk*u^H~pMulB}4EEBg?5-X=p>ghx-90=I5r-fRQo<QgihBcZk@~IXdn2jZoI8nF3verF=<UlF&t6blk89Pf-J%r(WwRIrBQ!?Iv?3b4uc+OR*a;o!gP>k{?6r<WBtu3AqiI9IooB;fpirM}_TWSFP=ZL&FN}GEetqP3Tl@>}AZg2Xs2)*GgXrK2!tlKnxR$8_P6TOX*Wll$6%+P8$j|Ti)z9yb**d<mTX@oX{RPcSb{lA2FnUM?m<0_GzM^&yUw-`HuH0s|w(GYGS}aY55|O~}4#E0~?775fRUa6{f^xeOS=01}<ul?Z1pah#$W-wl9h5lg`k!eQ8(WfTfd}`HrK<g8rBK8(AcpNMY0@I)Z`?SBVq1AK<uFoFS%>=K_5MdAB%62R_KNY#)!&L?ae!?MisaL~4JxN(M-xxOJB@YKQiHnY=OPn;#1&80O^}F$34sB05(djq78s5K2-$EF;UtES-E}PTRf!M#<|ZLN#E2Mjt3c;c9v+);E8`B^N$_~9Tqd9_1ykCyX{|iIp~J?`Lc!&5L<JP{?dx3<{XAgmhX_-x^&}?x!Wn=#qhP{}Nsa>S2UTQ*u5N77KXroh>L&33)op_F>JGiDRVw&rTjU__oSJWsnpfM8cQA+R>P~GDgvNWRogl$VrCg`F?Sv!LR?3fIDVYGl&_i|mev}&qab4)BTA`9v<fN`5;dB*gq^otj!#Y&A>n=>$%V9@)qN6l!buM?pl(`On=tL(wCJRtOrHPfHirhOiJ{y(#2YeH9)kz+Lf8r0=SPr`zg8k?VOKc9UoTEJEdA6^Hx@I@N6e7N72Y#^wzgWSXn9S^RE;X|v(jT2v{&tZ6c7QSas9MKZIZb6ZezAQDWhU*Xd3BI2b70FTZfCfWi}yxQ#mU}eG`|`<2y%evE6AW^E{J@rfUhvmBnbYSnO{Q}LCm)~4s0$yEY$0q5RyhS8yH)K%-MYR;^%kIiS=e<L5Ly-)Y(xz5q=d6#dOl@57<>$89TL$jbok$^N^XNf3*Mp#{&Xc+x9|}e72@A2JAA;X}37S5<EiGa~PJi#Ko|VEg|s?q8pe7h&b?3U|f=$$lG`BEF|)c?^3T9+=&z7MwwgjcM<-KO<N~skhyN2AHIA;bjx;R*kWa(Mb$(V`pLO)wJqZ6ae^6HiI=WjwBBHroD<$__oo%J@FhG|4VKOIfB5WWu#tO9N6;D5CChfmC(M?_%=<}TK241m*5Q>c4!5!#Z}zH$kO{k8x9C0eXtdIgw^A0h5j@)_)*-rs!4OT1ueLoGLnkA(L`9XuPP?FoU0nR=nQ@Efty9t}dV=%G-mnX^4x^LTulGHF%FMSX)u+;(h8wtTGPHD+&}3ihMkm5XWm0V8vMGJ2x(_w?VWZ^af<&(NVvlv+urjRo)n6U|&|U!xmLW9m&<2-#8?`cSE%~3|^L-k_ed_u?^>}~dONrhPWw}qp<~|{*``h&9{tgFmtn&XV?D}5d577wL_Ihh!SJK5G+q`|O`j0jLapM$f&^4R5-sY{hY3n6xeW@T<{nzZrA89+PciE1I{&qa{x8tF|9S^_9c6?b0LbX2t2vxUk-8A}tcFR6v&+b1)DK1*YHiE<aeZk5;R{gqFzi!p9Tm3f%$?v?qzb}~D$Eshq>esFMb*sN8h;RG>K)hD}>N6lnIdcL&^AWgTMd>PvS2y`wZ2c>Py?w0tb!&dznqRl}Z;Uej3CHDM7me7*nqRl(*RA<=YriK7s{a8{&_?Y$oGGP2*%S4dkJSAtYE)6B`WWY|R{5i%CHuJH*WK{zZuoUK{*6)E|E1&i$3%1Xal@~>;n&^p>u&s>D01@;fFkRa-|37h&CQ<B&wNDh*SK}mscrDNt^bKO^JvpP*8RG5zi!>HTmLskx&K1P`A;;QMZ@;7?$@pRb?biJ`tOOtxBdtyeCMBd=9S0Dp7_sv<nPx|r-nK;R<YK0{zU8F7p>dJO~3A@Uw6~5yZP@-68^Qp<4-d2J;{W9-1O^i`gJ$`x|_cziP`=GkeJP_|I09>JXIEgJ~Ig2uc6Kc>TK|x!A9**vh%M)cI@MpUw6x|yXDv2`Zp$7{{zA8PcrteLdNXlmS1<vue;^f-TFOA;Laa_1a8&;1z}Qo$Sg#DW{|wUfjS$gv%z;@8(V*p#XkmFw2#|<-EF__wqJMq-<YKSLBRJ<GW$m$v-WY@ue<Hn-S+Ele>;hWo5g(O+hr$Wt!=!(BzF~CRhyv4)|XBEQm5yW(d_KpsinDQ-oM`0<Rldi#|#s2NwE#TL$>G6HEBFZ*+WsPNMk11H+vy+22?f4E@^%%SSKg!RpB_Fn)G#S2QCv<M{P4dq%V_PSnBdPKOYS=3m1)TsK)3EX`fqot0wvE^Vl-6e~JzA4`04Jey<kvHec7Mr{NBUXY6>$F1HQqa5+6ER7s;i-X|{wkxbOM31j`{{e<HOcFx6c{EGbxOUB||jLF3_fP-PTTb;|`p>@&}m(FqP20o%eycmxXN$O@(|4HhjVYeK-WA6tT=f@gQLx(eKuggrtMWGR#B;!(WR*huVCuay)K8}Jg86!@}S*3yWNt2jojmUCkUC{%M0G?(p;sQY^7W~f4ZE7hs(Z}kGp}$#6IMm+3*CUPCVVphAG*0GOH5dAE22boTJkvNyPoD;LcEPFdsym<^p~<So>yCZoFp%f!-e8?T)wMG9EAd=Fe)s_{VkWK%y8R)jZbnEc!f$9ePieS_a7aHJ^fTuGXU+e`tzvmrMfIeN2cvlq4&%6MoMWvQO;p_Q++dF16jt-=0%m$1B3{@`L_P?!AvtBGA~!iAZ+jvhrbFw8mF7MepX*CM#!^uA+_kVP@YkZh8e_@24SMk_HAd&zgoEvm+oynu%sURi59m+_DLIfhj5vsA)hHLdaK`C&*!8GRyS^z)N5EXuxB(n_K%hZ`4Xj9;!h!a+xp8@ZkZ{)H`F^rw5wc!ui0j0`9>jf(VS^4LoY+eA&sl<hhmjwo0#)`H!niWZV-6N;NaQpRcjTj3AO|c}2NY`1K6(O#6hyp0)Up|h(&CjgTP&3gLyjg*FU2>@3-Zf}XT{?@N?mAds9APUei$I80glaoGZ$8=AehrxlD6cz-0<F~^$)pzHmOD?t<401LU@<lAW~tdF$wMXAJU)51R_f$qMa$8ON5hUmao(mx-8Ke%R~p_&a^pQxTzV-?X7yyECna`5s>zN)ud_!hmjDhH)k6)I0r?_b)>yAi;_v_m>q(l?&bDuaG7>Hn8DepEHMhoW}Nh4bTuBtQ<~kP(P-aH<76#(Rr|4d{LA6t%a6gDj`HL_pbOmm$^DH}Df08%7w`7pJP+2EG0Z`BQW|%7o_KMGBW0>+i}Ev;7yhzv@B8nL3ReS1?%~5ZLxwqZpA}i{$`RWOs=L`(S>i98I@UGVF%N4fxetK>1aF4Qhc#U8$S_-mX_BIS&L}bMpUsdvf=KFO-$ZdDer`VLPp7fux+IRtOEfAx9Vhs;87sjPM98bVIIyvxY%7N6GHja7PCzU8ra2~tWx?5l4hka>y!-hBJoMT7`XLA6_}}zXunE;gpb-i6(c{dIh`ERml<eh*m=?t}ZShSsO*1cKX_+Bc5*a1+Y7~frYqN=CuuZLjh4yl~^0v{w^t*#N?>uQT#NN$?(&d^K`P|qv4wjY!NTU@C<-@h`&3js9<aI5|j9ox~B_O54e_#crMlP1+KbsZEY~aEq0Txi#NF%u%VjOvCIRmlyAIU(t1+uxth5jgeusC4uU}1uTdf66fK>)6M!Op4%FEDkq`iJE&mM{??3z-OsO|qa_EDRG-tMGprK4Mum!Ge5(f!PRV5)jnb^+6|o!A)Fc_rnQXLA3~Lq8=hzWI{du2b~3kJ~y(;M_QAeXjzWJh2T6*QiyT+NeT&>5J__aI&#Qn^+wOd-!-uaNlp|5K+7*VfF(?DXEc~yk${R&0raU{4j$GvH-+HUd`$o%rPb58DoG?V_k1#%GSa+q7yhfJEa6cwWrrI>WQU@e-_9m}WHxb;EX}<J-p9*h$-B>CWoG|Oc8>@0PvP_mdAmC^a|=cBi$ue(e6*b0McaV_TI2_Tt|c8PZ<dAJ-EyvOk#cn(acmaK)h$+YbuO7Uq&t)h)<V!t_#f~$mJKU$3W1W(el>jR$w3cPN1=hJ9d?O3C+@M2Yy;53EhMzH4N4;c3<j+mBGRhak~r3;ymC5d6N}9}7`oI_l5|Qrz$J<WneTFq#T~LD4=+q63(hAyyg1IpPJJh_-GS_eh-3!@^A4d>L6-l!R&NCHqSY}W8zmkg#vVx2MrBgsuZiD^$WYJw2$GzQ<Lbpjd{>g#fjSdgO9Q)psxB=oaT5IsmUX$XEDOSO(MjUCVI1&-Gm;SU&4>MvQ-shTB3vx|xF3W>6Mo-~$MLWmQ#U^zyw;5N(@B5KG6+@CZkhib#)~PxgDQ-A#K{<R`B+?yh#hXV@h=MoCWW1r=8-T;zX`5dH#CcW3jkRyZ>(<zGZs5I_)mI0Wx~sFw5=V0;*ZY>f-z<U5hM<9HoW9fYQjJr@S3wnR_-`D&>(OeOC^&#g!4)udk7f2PgVZ$aF6O(iY?hxuTPE+OgaI{o*H!Z)m;-W9vx`r#e%F>hzX3IgG-Bux=6&h7Yt{E!FW2!B+SEDJ)T18nqdT($3!%#PZ7es?-qp<pKwZz^}sR@B|ou3hK-o4NK?RplR4XqA3(>8vaJ)!1?V!#Xu+l~<!Lw&FZ~CkK`n1WSu*4}=rl0qlkf(ir6gk^%Mg*~8K5FR(2)F~-`6sn)j6cop*bU?RVJ7@(To#FXmF~mni;80-o7N1{nsBFcl)uB)R(U3wMo;-jb~CgW)e6uI5SH3^BMu$tLBkok|A$R!%ash*n}wpE9()LN1}AAHb9Agl!_S_DeB53>-nVFdI;Q{)08>E*&}>Wz{JOziQrSU6!`U)aWXXUvd*u3Hl7*Yg{WQ`amjL=6R{XBOMIbxM2raC#ry*%81EJ_2OKJN5ci1aOh#8es!ZauD>_pp34kto1dDa`BlbbB&6t{N93m6MSa)dH)iCY}=N&?LO?%09-nxz@{Fh<Ty2NK(B9Ua3A(IzI!Ub{#oQDgmNTSJe*&3qzce{@INsbhgCYENoU%drW-bHQzd&0Dv?$f)j5{m`W<m?KGQEhfuzW>rFt$YXPgex^RN@vMiqGWWy8GQQa3!(2h6eAUqDAbvV@pp<755|W2Yx42Qu{QF;OX7ARe<m;}5VTlwSq&l^^}A*$Bc>pePiuKPtxQUs3qL1LHYH9rAr9@<@w|WZ>KEJ^w^)}YY~g_y3qH$F9w<kNpxZ=nY5(X55EQOowJt^z5%F@ldq&V@(DIlZFr`(LV_BY_T#k>d7NpHZY5dCw3O-;gAPnCzGv#oq^Wr=mUYd@jSuB=j?|59Cy`x({Ij2pfKOlGbxgt|XE~(Gs!lXWry`(-~st=r<JEY<%NGr)1fflZa1`>`{mjvgQgwwKQXt?eT$4s=Jw(43Kn}&6cmSrPfnpEi8%^2j#a9*r)0*D(`L=OiyLJD#EAwMvN!bwj)%{Xv;pRwyz8iBdjAGQXfQa7kRiKV%aWRc{2*6Tq>`pBT_s5ozHb<5_d;e=R~+yPq&CI@*iaSImbxDkFxlh{bc6vF$(sc!ORIWfxjgeRW$w?WaIRf@j_j&8{db5WRGCet;i#hF*Wty#@ClWiJFb>`KjSy{gPc9O?Y6<1uQD8Z$dUEKuMx^>IUAnavT_bY{dz8l#KKJ34H+kE-_BluOEIM6=+B(!fnHqSpG#`GL$O&I-ZZM#6~`{m%3NlKXN%zgbY4VP8GR8+5+`Hj70{sqY>?Je_?D;L-wR%<)1u5AN1k7vbqF3u8FUGi&quTUf3^9|Fra)Dl`Z$E)flJes|BfcBpm^i^l%@r9NCYgBOq5~~UE(BTv(Wd&k*;)Gm!t_EnOH17#E^^QV`tudLRQXX*OY`kvgZLPH2L?Q^pbe=z+PSJxWmb!ral6)53~`R+7uB#lkQ6m0#xxyr@I?<%nX`LJMD6)m=7=#kJwLQLOf{-0BFXZZt0D~zM~<zEmdc4>s)k-Try_=_)fO*Y+C}*MFukBGsvh#}HuQd2*lgP6PMp9ZjO~@s12jIIm_WJZ{%B}fdZ`MT!?AhNE)D;l8a|bn1&;}Z-k9U2CSkkgG~^qU<56MtCb)N%C<%K+<={OEi&5dlESYN0B2MthZ9(LT*NOdKCscJ7Pg6)CN-E^j(OikHk673ti#^e<p~tdfp1omzA{`|Kuvzm`$bP@G+%tv;)YlNV48Q5rx<e|u(x6@fq+BQDW%?<;$oO`gmL09jM)~7RH3M`oc8d(av72R@Lz5P@XhAV7bA_N|3)c2%aMU1=>~0H;yI7p*lld9+CsbL$9i#zhc`IG%Z(koL=Ptw1p4WZh60&fH5>amF{qb<FHFv?xfU<*7$+Op1M286f7SUP0Ba87OKheN^ssZl{lzzEe%MvrMd!P!6v5B5*!(*e)h8eR)$^ZLbRgKNrn^*u%v%nFc{u)n4f=sNI!`hvQemdzu5EXs5)pTo~9yG0e3I=G$=Y7qKVpT90hokBWbNXt`)rNfT2Cr%k-hnYI3}1V*!GM+Nrc2B74yP62dhz248k3(7U1X!#g0D|%U!z#A<yC~-GJm9wB3IYhS*q-`d5C)~v1t>LTuk21)PL209KJoD7J2nR?v%E<NRTByzs^DOu^M1J`R|SX{Z;?Yzc|%h?avcpo;Bz8MM?BKIx>h5874fBrynqzQ7Kfvd^?)?7vZ_At`ktGYD=U&ViWL_1cn<_i+D#ivz|pTqI&cGy}<$vC-akDWeMq~sj-DWs<C0(IWWMe*Mot&1kKM$I--6<X!E;?JT%;|vmTQUX7VRdBlM)If1!F@qMRuBk5fw~@uwomYNhaGREK)ykwMQzqE)uYYB?*Tf&3P+%NjsjR<}BwrDq1hTLx|y9(goweVXBlPdQrTF(b|w37>$O>0nIz;{kh~+<*uKFJJFJ55mLO`@y(Bj-!IU*mj`F21yy)&;o-ohgt?*$LCE(XY3zE2hV8<c!cXrx>Y3#u2hLtBN$H?NRq}ErIvDtM2Esji4y{BL`s|doC9vQfrjJd(why1hu5@)oYp~83<T|27d&$Kcnd^uvtBm_2kJF~0U6W7G<$(qBFF>?kF<t2yq<uNpgW=hxaCDknotrL*vyIZem7o!lO{aSG1qh&PNs?6rR;H<nd|7-x5;bu;8S*V%u8=B!a=9mYHe2&cQLPmK`>t>JwrT_YY`CcgpkUB?-k6HsmgkVeGrLeBL!M(C|Tkpw<OY<!W2Vna0PueOhT<REgPD_c_jRkV6ia>`be9B3!0cVyVDxdk{M}+B=VbvY**J5jKxlkkFt7k)c(6#_ciHpd)7}qQJi6i|KAR9A+(MJG)ZbRoB;uLxm<KB%=b0XK&Sx0l%>Quz+8-wL386yCAF|*4$h`0OLK}Qn(+{ME+ttl0K~VBOl4+vuHrm>ox${GOu`D6ZZtRvQO@*{&8Z172KN<UOJa-Ty!xn>3N8K*)#5)jwhB@8k+s++L`Eol!<z8XSKk0Lw<ORLbt-x5^5w%Y7GcYV@zD&UzG4{ZcEbLzTVaZ<FC(X2wvq}p-5Yy6nTA-h?kf8%Ng^40)Yk#73OK2iNZ5<#bgl+aMY(DAQlp5@92T_?Zo52ZOGTn9s@n%S5x<AXF{mJ%(YdX)nrUsP`nYBr<y$1V4!Dg+9h?c1N`Ki2?BT<~Sx)zC=7zE~!vZrtj1^yL#GYQ7o89oW(qFMeAsXPM!$wkymaiErlD~e%iu>@FQ2AJq<au7qWVm8H{^R^IVvkrY@Uro;elw1>$IfQ`w$2M~CdY>L^JEl{q>K)4U9`W&R||ClFCs0O`vvbV5(diyptVp6M|{<wkm4l!;S8Iit}|&Nmkh)iiW)%R?-L!!#l=$dnG0aGbC|Lnv!%hVHJ~+Y0V~9G3<|Prh$WigT%(qp!Xn`!5x2y@;ary&#F@#Ps=m3gZ)*DHS9*iLbI)kTHae+52BQBHQ#i@k`FyH}#K`(3eA^q)V!=NwZFhr{3nZsC8CQwze7M6mY?$Olkic}07%u!(5&!MG*KlpXPi6v{Qc8wrHi-PI>7YN%bj;p1z3SG2#$}SDDrecJ^6d$&X@_;EbOO&?gKH=Qeti84$@}FWE6pD(bW+McR+~RoX(%<O*Q!+&I*jdN%(-GqlDX{36JBmYCcGv1OX6tK_KbtW3pQd0bUNn2MU9?~<&y4Z8aHVqS79<IftbijWl^uxE3@DqX!{?yYGNmZ8QYfc@YO2w<PWLj=Ceu#|M$BhmW+UfhH-jwaRG;SxVX`()dnH<nYt!$&lIZKLgyh40vB4*Xw7Z3wxrPwx6zGUqhDOvy~zr)a2ZwjYG4H_6!ztZ9~gJ2TQBMYIr`$B#?65M+Gub%oNfuT;g}955V`F`)qSYB4;%Eb==#eBw7j}2?aAM<Qht9*`7J2r7yLFccG2}&k#JyCeGWQP;<bFQ1mA{{r+r~Y>Phvf_Zr)vt6#7|EGIm5%-wuDD=($5Aa`6mn7qCXhI$n)aIfzWABWDT^l%9g6fY5U>+fCQ$T9BwFNMpJ8+~dB){T4d1ekL^np~EQLQ^K>q`6azao5p3WKC%!KoY!Q5Kj{!#UGCB#h^iy$ju#h%PjXi%H2E4ElZ2|!|8M&W@QjSt=3j=anh?zwRLD9$LHpH^?u}UX$G>KNx-dMbmUQbIqj(9#@*s8aJ$&n7vteW;;ThdwX4>~XK|QJyNw1R@wf?r<nx#4Hg7Q2<0s(2NgAbxCqYsL#(>o0G}&trq&jV;C%W>~vCsVa(}Y@{XA2D9*f6vrxOorhQS0T+UX)%pFwWlIHFpZBijvQ|3o)(K-9XcnYj$%@RE{L&4YmNJm&4J#q+8vGB^$I5p*fwAqt?Fl{^3rS4=dSwwt-_6J8cPJL+r_L8Z9>kbu^*Zbwn0YW$0%=qR<!b3J<iCt#CvYvc`d3hEo#;#y%npOd;w_i`4uY8X6{}*#!&87rp8dBAZhXEu`R%xa8*CjMB`c>4mX>O<Ub8*-p*<{*nt_b6W}hqUbC?lXP+0Wq8DqG59S35K~ZMe5!Ks79ldm2EC{`=9>u|2$RZ}C6}QEE`4mzA@@&VkvT_Hd?%GZTDKjn+wTWT0q<<&tNp!2tMXjojY)H~IA3gV4eFFzD}u59z%%xp#OfJo0PdTNet5%wiw-=Ez*4fB?9_B2%ydW${A`M!Y_sy_KmYSJ;?uyi7xW9tnpW^^|J5ro6~<jpqMFoHhw5l7MTuI8?x>44ro!XyXG!x@;f22i<Fg4_>a2^~1$EY?f1vz2tM2Z4HUn@(rPr}Rs;gXC#Y%zw+gXPii_J6J4nl3sTIF~#owP(711FidzyY@9yC1(j06SOByeoD;#*7fpt(A$XamJ*H8X>TyDo?2pW+Wzc1Q9%W`1-24pgoCOm%APjUgn@%NgZ?<%Egu+aKoe(U9ZEV*1>}-$dN=GS`hnVcsCQ8$>6O_Orq@IxEf&$#W#wP=<KWmeT4>F{Y`R7JMLQzF&>(>nxkHLT;i*z8Alm%yt<;iSGRk2Zt&J=&H_3isUOLm{cF<>utK$AF0P)S&#FN`0+>LceBovhDXy2mD|wrwZPom?YTwe<hTqo4x3pFF+p2#{Tbq7co8QvbmfzOax3snGx3&En+aem!nhmCYV>`d8F@;=$&WC%30H-8(IS>9t9Z{3%y6EtbuiFXy&~RJ`I8v2bk!frby#@vWAZC}K**Ych<rVWI1TXUr`oy1uh&zN|<VJ114JK&HOk{`sK{x1B<M|1J^2QeX&1!HGy^>zQz?}ZAZqy=<aJE(FR@7TGnYGnnls0NGK;WSBJ^|xoCb3i)K_f5tpzG>^sBq^z_S|_>f8RlCHtd|=ks$LTgr0xN&}b#n>#|Nq;x5pS18ZZDYBp4{0n>{M)@xWi!fOb-ojpNn)WEJTtpE1F*lST&4_)OHV?Nr*uu?M4C#+O@oK`qkQlskE*h$Zq3~N|Cu3^Ez=+ks0ehyNQo(|;kz@AFI#GA^&8;_Dc@72QCLXO)S@K8Nm&PRzJQr50tw+0&DZ*jLcb!es8nuY~FLPK>DLd=i!raF7w{{TQ>LV3vSiPrEBdn-B^$=07kla1{Km85gt5CzAD7*XmuLPUo>i65gjgH&xDZ9HXTWQR6gv86#spB_@%Opr*z<wz}dC#lsFD#t#0G<Y5gJ?{!5z$4<vF9p}~9X|kqJrr0GJ{5zJGR-E?9yL#=Bzz2;#&*PlBhEk=F6fOvDAWxI4JA=7yxvk!WBhd6Uh$oQ{5euiS6Sa5ve5_aOyyvjJ>-Mg&|f^ivx?!v%z-&Bai{1|kXbsr4oxe4Qa5o9Fw+5das=@P9Ln&*U#!NRdk8ynJCy53fM{t%pHRY$uKy%f>wZwXMZi}WK)F$gWl6D0Xn2O{iH0l{AZ)>Ez}vauXvwxEcaJ%+B*1`+Y_T?xeJkfe7p$_>$6pw1s1D?p@j%TrQN5*!nr&j^)LnLAWd$io<W50G8dXJ%5FGwD^~DsH`F4Idx)WMcHn#v!41gpGAioV9d~u#-1#^sxOo<k;e^*j~D<d&Xbqa+^suW0JL^K0>HM$BlUFhB-hGT1ThTyOr!Lz-=Xf&}L$Rn)56B?pH)*VB*Lc&_Hgd{~*MyzJfT>L{2N42S)RBzBB9DO8J&<y|Fbuc-DEv_ygsND&ud1!F0?)dFIl}n9VP!EJ#R9|23xM;c~NA@oyVfTA~9~PSB@sER-M|aRl5TkXgRwC#5IqMaG{h@(-fZ?$|WTT7B+KEvZ{BqF538Fvy#5<?&-^qLWv!6AEK4uaOi?RuV)(sE&`4G;L?ZljzUTdmIQ9v&g8{)~RjX1bKoe*auwSFWJzU&W&H>#^k*zh<RrJ-ZkxTvpMK;TT~SZD+2%s>_Q{^gthGDnC8^b|_oW^+1ejZN4#;%;yB_B4DAQoa+x<Oxjo5Zwf{{K1vz^jpJV2OfFxFfk@z@a(4l6YFeehV36%e-_)VsHkb*BNS~4u4fqqs&^wtVEeFkaL9rQ0rL`R`R1_i2p!qN&dGCk2Ekn%M9bkhEe^J<)HPxE>uPROvGcUG>p}O1&LSP%a7aBL>FQ}TecdIASRI^DEgBY4AH}+KjMv$u2KD*gfy;~P<q0hblm<^jsYbJ*XJRvCKo-|@CZi--S6&sgYA|R>Bx^=KB&TS+tOMteNqA?g%4ICY@10}CTj@kDV$KobqIN8OB_w4@9E4sBg~XIekw)g?lPA>1nkTl>Zf8j>hC_@64=Td5b`H{be2f!1;~u@>Ta!XFX=hsw4=iv2QXL!U4lomacLU5(jdpt9mS221fC9S=;O=hQx7c;;&w<$>nh9YuM`0#2YC~S8Z|Q5E7r^n^+rz`)AO8@1lsfnrlW_yJACB=N1PY9(@~-)LAbM||02}hn69*pT#FPp`tazH+ry(QBlEy5<Uo|(gfa`Sez1lCl3(2+QWAmrs-MS+3hIPxD^Omsmx=*xtfJyv6k*Nxz`0{s=<y-b@$XIH!aScI|Aln%tqMVOlqSO^RQB6jTeJ>0-Sv%WeNaZ4_IXkrmL(T9p@x^NBB)XE;xIvHu09(y~=^a`^+{kPa^69q6<>=Upk7@ac;JUzbb<i!F2My&`o^92wb5-hCHCPL;O{xNGJpIEGV>6Ir+6^u@Y0abcr%5t4zV^x-6gzrQt!wY>Ei1jEqx9ZH5t|TNCTod=J+bCx0ZdlMErg^R#C&PVyXu{Y4A)|q6}c`n%t^B;x}2sc?wif<&HGm+@L5-*xY?rv(+!1T3C4<zlxZn6+RIK%gRbV7o1zd(S*S6MSif>;rs*B?x3z6>g4G%aV8EZ-T(tT7^Mj+XVu=ySY-cpHE2e&F7%^FphJXx=)y~wmS%Thq#tfJ7WEc<Fj(Rp7&p<{rglq}wgwccz?qh4N-kBii851j;OaPllEXb~<fMO4hK@j9lC;j$}gF_M>o@gorIu<bT&cg*b1eyuLC{IR%GWsa?r}MG2iLgXe=goTw0Np%~2jh5R6t8!}V3Bt*Z;oOH!n1v)R1a@)GpW&j{W$+$Hx6l(`7S@S@^5LzzpZtrb+~{0zS{ixjqwF6VnYS<PDiuOIkk_NmlKv`*}F&po%mEvETSLT8CiMLR8t|WwzA!$!N5kgvw@hjk9BS4s+C+4zeD}(rY;8T6w^<VnMvEBwl4P6!XHTtUk2TEQTul22=_Z23}kwq+d|?KHfjaz&0y%e^q6CC{Pxw$_v!t7JTc=X*kdDGcryp<Pes!W-MM}biI5CJh;Xp>qXPXQqIZH?siK`l4=J4(lN5c9jIy}b8BR|I!&ApKITsY#jRvd^kA3(%j?1rMUd&Gq_uOf_6Nk<a1!x6!UG6RREvB%D{j^a)U2H?Sjmy)~__EpW!gQt&xqe(8j6``!2%V%`V!&c-z_>glF(5Lnr<PY4!~by%vVr!mfP$JWaE~<Yu(Igsf3s2$`xmF&1547_O^z<vOG0i+bPZ|`wcqWAKj_-Vrx0hyqj5(IM*e+F!D_Rn1iL9$iFS(~2FE`&-%Cf2`1ftbGeG*Z2*4v7HU9f#FRK>5=2b|}dxPd|T(AZ_l2U-{B(`kgV9MP{M@ajAy_>%k9zxc%?JjL}qaZN$@VLryz^E-e`o7P(+t=<zY(XTP$x-#r@ju<}uPa_uPa4gNAfPZ6xUR0MP!I^TXwK%Q92kMIsI<F|;kd`57cWk18CL83El8gmLG@N2<Gung9SK{J19=w$FPqSYSQCc3r%!q5o3DO8+7*Nq($M>oooZdZmm2m~Y#2*!CacIGPHQllCf3XW)9C~EU^qjlxdo(}O)0tT<j~A7E=0cvGo%!vh~ZScdRTq2|L)a)S<EUFVSK4R>1QLbC`MuP0@R5ZX0^mm2^W%ZAu$(xW?JHfel>zgATBQ@l>=4kPmR}|M%s`(J`%OX$pLV~dX>~2o)x@(CSl+#!NoEg<hZi)JtPm=@zP;6>6L@B(+zvE!-1j%F=s!<*FfI{!B0@`Z$-#;FobwOazlJwm9pSIR#vdt2r7=qE(p>!<}Gq6uc}wM7)fF!QVYq;lA8;rsJ4iQ1ue`uPYX5h=ELe1-n_H%TQdzzdeVY0g~s3w?PUwbcC0C}DT?V}Vrg?#>$|N1x0gkEuw+8l6#L+#Z>H(AuRcjMbSh?wDkSkL_y-|>wql8QGws?<E^55`TO0R~4)?NE7u53tT!UpA6qB%OvIqo~c`ObVx{^aErrt{j?k)KhH%_uNamvlnqyQIPxk`tklkYKk;8Fe~hnoHbjNbqlSy{?qqJ3J9Sh|&}WC<BoU5!MR>EO*~Hti}ZU_7$lJnq9)8zaPN5TZ^nZS@C1XVB_jNishKT7t?rbCAC}TzSDH9rj~&%<}8%^3qF#mS@(gh*x1&B?}J4`%S{<;zpAXIGGMs3+B4Rw|ENPg}E`J22{#X#@s5%@Wt1VTl$5cfqH~}$)2jB(G2sW_^O`xijhvoFbS{t&N{qynK8NJ%&q>)Y-Y$WH*<p&#^%^RMC(5Up~65lx-E@aE{~7eUEW#I+b?pu{yj5hSY+D2Yo4@!=}c+4^cvWJugQ$&uE)v*X%>?o6CB5^If0NlR9R%`v*Uq(eFp^xf+-mt&}BU2EC!=qu#n)yuWE2U2|~^aP}@N?Vp!^>SpnAjw7e}`MyZ8$sxwM#xalkLS@BHf27T{}kPaMc)~2v#8l;Mc@W<zH%2Z`4e|qTLG;+15D8dhPF1SQ9TmMS@PFydOCE9O=g)yOb$8~1VS50o@oBs7hjU;G=aWLFwkDt-v!|eh&Qk1n4)g`OM#yde=zuFL&?e(YXD;sin22A(Ax9Yf9kz8h7Hcgjlu;`JCWSg7~S!N>)v`v!2f)SL4%NNt{t0@=wA=uu{h%IeLNI*?uzBa*=U2&YFZ3jhidalJG{b&4E{dm5C6U<mzVoPoznOCrMs4S}E^~XJ!dP{&~?_;@IgEQ3wqPjG=W-F2KQz3@69&8)6Ix&Vd$JO~dc(BI}Z`FId?oc&6G1|doLoCwYg7s_a{aTFewtW8bxICCmMf_zC-Zr{2!2rG#MEVzM_#dg3HKR(Fy#{~*{1Z<6984Fqv+>n<dSl*lG&LYls4zJwjpng|5Ko0|kao=<W4!&nY&VhPQ>1y8377W}E^QLWBH3E+LL2KeS{`3a9pW_l$13KROCutIu%24^kRIQBNS|LgoT*2ke=ED~lGg7i-}!j7lIe(VSsg+DxZKOWu+<N|*C4IU@e~O`@N?Hwgq+1g#i9sn!|||zhK6Tq;oMJ*O*67d67n<33}Y>w8mpM3*CxLKLZ9LiHJ3&&E{<zqZ)j5+S>VDBjhuv1<*oJl#~F6mo?VU89NB)VisQ2MM*rINNZ2_h#0Pd+Lngf>sd6g4B-wRt>yLkW{$FpF<6OyYX))|^nPh~HfU&=O99P^XTXbL*=uUsWeJ3}90#tmEnoOj^WdVvcQ{~;VLfQ>aQ<HibD?NKpckH+sR3Wp#<2pSXNg@eOp$;>dGbO#$gmqmu--WHx;80pZrJ2;{QeycIrW{FD6yce@)2*xS7APB*5PDWUr2Cg4>(ZBQWx3}NDyKOUARxudd{|F7jJ*z$4f8P4^jhW1vOC5-&#gUO3byc6LtK*=R_-9>(kiq{RmqMemO9qBj_q?A#)b>rQI3k=ZbEf;3H7Kzr@e11@~p@_Jn&x~q|sZGi8wQ-qUk={;~a?;szs{#A|kXNxx%WHt@&jS3Yy!<7s)mE#ZQZR>pkuhN2Mj%+}LQEb4Oc;5!_-(Lf=Q6kArtKAyk}C8c=N9tY%k;LROTK=R&XPE-LCFa<S@$juTf_6h;aZ4EW6LfW2Cp&eGKY%kcCH>w5QAx&1EpRMT7Tsr+jyy^rlifw`udI%+7%<N1UgZ_*SZd%B<xrOvvV+K5jVe^_y&mA%mV;$P$on5Sv(OlM28JY_QvY6`#td}#tgz|S@+xzQS2D|}+oOdt#~+~|%t1mLGxOdOirIB36KE@{n9#GYSj?{AFYH926KP8d{UJDW(@d{D;LWMIqt9p71D;6Q-S3|WWgCv0%&yUFFv`Acp@HVm;A<r;GBSlxD9oBmuo8&PT<VGN)QVwRzm*#5{dc}>?z2a#655LT4mC`3xf1=*Za5@Jn4vjISGoUZ#xzdeYvg8ORgAWP`Q1W02wLKliNhp;2mfON%FA8!k+b0of_7uQ*cAMO}AR3>wy9FUNM$b#`eDX}ZUhqj$TbgMy-8hhF4B+K}88jZ^aBgSopOEz3jIGvA5@PSO3RgIs|j0}V?q-q2cWD-2sb}XUCO4^RCv5K&M2nc#LPBMJFVa$o(gDW?|*tA`N_|{nqk+aMhIehu%?V;gC>EYSWuQ<W3L)20?l@t|7A{<)nE5RowM|e0MX+%;<sAB@Y$(1LJ!Hj+RjhJCNh)S0=@9uwi{Nm_sfK-uiRp8W-^{j1Me>%>yE>4Vg0wNSZF{9<-X0s-8mng-B|Hg$MY43)tajLuP@yX7q6H<0=UaW{Tp?PKz8f=F{%2+Cl=91I2TUPUuNH<DOFdPZuM+F+GrOez^3AM6BdCA<(&N<tRZGmaCa<raCZgpxwi)SAMZBZGESjN0=%8y&SAxDY{^9&(wviBSdnRdsy$U@ItKy}?IYo4*Iv7(_<IiD?|&Dj$Cmi&YXTlHuu?FCw``(!u?_W5idNWMh2tOA*w!&WhsY1n|v0P~=AmUtj-z;TJp^Z6DIosF-GC7W~+*)pqUq>_F`L=ee0ax-ipwFX~KDZWYygr|NbC+QxkgrqUV9dS3Sg1UQ4l7ca9YGrK>?@~ONq)7PwVyppytvEW#$rPoM*;SLM1sr;kPm^hwB#$^BQ;g)}4+11^Iw9~oT_&&-o;)_F5jYUZCgA%BT{d=8h}1*`Th>SK5eRD99R%w^0<kKFk(fLNFePM^iTXGHT}yKWaWjw&jJ>QT8FNwsSGQOkl-$v(x1)|nv$Jy!Ga|Hy*((iw2KvidJ{m}kD8qVc$1B;)Ue$v0z8dN1k@Pe8(t_==sUu;RQm|&H=cwtedP}AF32V2O8jHJZT|R}P8cmj#hdy@>4rds1h(W-K#R%>5FIVW#c$0j(UWUlM_LG5P^MOU@g9w$A!6}Psg1imp9i%CDg<y=6Z|KIkKBRV}$5c~cQwhspp5}w;c}XGk9`Lrz5otCvy_wMFoxHfXLfBo_PtZ!URLE6S+k0WIE}d|CW@;@)axK1@ZgFNb{EOee^6y{y_nd!&;<dSao6EPke4ER+xqO?;x4C@Vi-G#(?y+bK_hr@co3u+3*V{lgc(l+Wj%ax>OQ-7*$JgHt;M!l|KTlam*5yit(?y}NiWDmaqr!NN`H;Z3hSw}Uq6FzD45t<Vz&LQ<`AftM-SsWscf7+Z&@)Av#_XB5Bbv*A<}hISfC0@Tsqt`6#-)Rz_T5a_`1Iz3^9@Lcs@-Y4GOkvqb0@HNOagY)Bmh+%mE-q5e@noUGld&w(Zg4UH1<MoUE~Yjqdii{ug2>;y=eD;9)iJ=R~TX@f?sy|Rn>WQBd==Ct6$~ShVyEkYqoB`woS8}&MR$p%Xy{EZac3Eag&5FPfgm)jp_rEu9u}?=kHI$A4btv9nE^)TE+85s;SX9KPEOIYpppMhP-Mti~eaTu)j0;y8{}socRYUhkadOckp0Eu6bDWrpJg6H8Xc|f_9K*C|y4XmenP;8?a9io8oE^Bj(zLxy#rz3{okY1ZSJ%4vf$`abm5(ARaUfix_efB(YR#^>ry^0>82ugEESB!$}nj!Z|tE>C-f)Oc<M1g!zW8$8LXGDR7x4>;XxX$(C@>6AqdUKGMkZk8GSB&4inm<99iF6S1b6|4VK+opM*|Ig?K!9YXTH&(me2Bb}<<#JbB@21#;l#BrNw>t+QXcL$L?;6_=(b_8sduYpjEv*ttn?#kBq?`Bo)-+&9b6+OtbMjZL&ZppH~c<Uk|rEahJo7YWfkE4?)PUOp!XKtpYq(;~LHUdL)0COjJk{4Q$Z2882D|i+RpZiu&Em%eO?R-g~;S9<fRaW6V$mcu`1jDJWEcvUHocozF3}PDLj`0?}II^jSN9CK_OIzw!T&`A#JIC&svxrm7QY6Kl6{Io-N^JTw1u2<&3O9aHD1c@1L-(hudn=_#;m$bWpK#FA;eINR_)a}m_S5aKP{4&bkWqkkqFdp_V~?)yo+U7kemu;x3=*$%P<Wk#!0R0JUH=JrE4h{|;Mu#RZ_#z6;!1VvYkb8Zc-~c9`2zycZ<l2<Vod!mi7QsK;z0pmi<r~Ti#c^>#j}7aubC^)LahB%JI#Ww{j_mf1jD`_#rdO>izGn(O3`H$eZ?W&$u8D+m0G@1W^pI~Am}F3Wi!2t?2_h)>YA|sBl@$;k5zfBp62g&aFemAg2Hel(Ho{C8V8cxAr8u-KNS_R+*ZsH{7LU7jYVIa8vi9==7a8N)*i%a@2T<0=^yE&brmP$RwoWdmz}@!Z6n;$yK(SjI+}FOACW*64-=9e%I8nrulw)g;iJy1+j{ium*@M>Mzl_gDVO>C&_rb7wL!y8He6>j`cSu9KeC<8UK$~T3O5={!(`+yFf>pJH<NA$w_=k^@>6O{Us&#=DJMzWyt@e<>Y~M=sime3byLtd`@j73K6o0y2h;0I)iYY{WA!Z?uOl-5uG>@F=}GDx9~Smia~wVR>*4$6d)xfm=HYv*ipi`L&E_dO`WxoepkEs4(C!`^fi|&u<L)56l+r6PjcMxG7---~6CYm9KHolnMaLOY8D_%)k~8vYAaJ5a!)k>7IFO6`+GN=q4LQuYGltR7WHMmmAgAtRHk5=P*Kvmqt{mP*gAQpjnmi5W1PjZ*+PEg(`(TlM(eEbN2ugl|={#G7&;-X~<OvN%>2G*5rcWUJ3K+a~CT5K!Coj)on-{wI>Y_QJlXAB;ZNX?5I+1ZXDJoUXmO68b$%PITt9iw=*MoxgHgdfdm8$1^Jt*jPv!K_(Z0a}2n3@W2W!oH%CRg*ctRH}fDr^_cX}{;R?pZ5Qjh$>4^Mh5z9~X8oTT!E0$@j3bd^OhvCz&n-<_VQP4XVg>$o&piwNuOYyRv*{o0&L@Gl%Vq7c*^&7YjKEgmw{SA$-%q#r3!k(^i{um*cPmS2GEEdLt>M1*N}!`6hTA<&gaQpWi+EvH9%B7ten3r(O!y%$4sjj`da)Pip@rebahj0ZIYSE7H48w^4&a6%hbvhO?`9fNnCF80N@X3i3iOy!kgX0hW>+zWjKY3%XV?W*fOz!pFAgN$ENBr(rrUGkmOm@|8QD+<nI#@x}BsoDBx_F>T7caA~EalC~BCW0uj10d$cWpLGbg6=2B{(VL=z^Bg*v&hyat+W2BF<k(cW0S=SRui+#wp5^8ZeixhnzkBub{`1_);drdnC7lI-r0iK0S^IN*>?4fraN0kcjb_9M(do5wW^ib;eZWCL^pp2SG9#M`Yra<toB8zHI_?Md>{bepQiuIY`t4m~gTYToNvZ`BUUgka2MRJRxifeAb6^FM9JJ;J2ml&1&$O_6=wq5(*Z3sZbDeM=lBKo>GH&h0)VYzMY&;RCXZ<QM0{4&w$2);$R9WJ;0{vt$Ma{<)MW`ajqcMy5?&lB3QHnHfLlVaHT#$lHC`<>DVui&%uY;6WiKk431Fo>xFXD@NnE5K=X=ct4{eQDs({EzpQ?!A`RVX8I7h_QKwdV5W$FF{Vf6VEk9(XABL!`@`FX<yPT^ycrioLuk`LFG4igDdYc$LhtqPKYv?6xIaG4EJP!!{rx<fBHWXVXK=#QD18Q%;?&CJ}1Si`=TRM8YOd84)ZE?TTutUYEqV7n8||kXaBziH8@5#P1)+(|r@lPee2uUQtcAOSY^n_`0=tE*5OZ>}~y;4RnubSS3NV$(6Msayzcxv12y%{<eN4rfXa=(OqL>wW+I#E}S%H)YGwqK-T<&t!blyw~fYGJbgBsOyc1b;;@to{TSsWdQI8&j^j!Y{=06Gs01?z$WkoR!`=wYkwhHr1|cW_GfhOdk}KTO(SbzV#&`_1JI>#z$3E9;P;5GkTi3>AOdPHe`&8_nH4AhajI%`rXT!K_4uzGtyo2$I1wRt>d;K$HkLck%e>N%M=&`KYOC*->Uu}f;ShGoI<P>BTs8doz-1o2CGHUz?RlorbEr93B`L5z7y`Ao62WgK2ZIDs3U{67-IIw0x(YKB)E8BPMO0r%X*R4T$yt@28A{@rzCzB2u$BMIdO`VrSKlb%~)iDeO^YT<e79YGkdL21~{)uh&Uo65OsD3FZhKsQmM)Ize4K$CVCAt0zhgLL#Cr_SOdtU8^W3wh#kj0-qB~l=L@c0UcamOZD6Y%%pU+>+!_w4Q4=JC5{AU}Wo>tFA!QfA%z=j#sc{cfs*-FvIFD7*Kt829e|C0PIF{{p*))ZDYt)wn<4xS;wnDG_qIu?89a2~uJqB*b(!9>nF~JuIm^;)2}K+SCwZC3b739we}o+Al!AJMCXHP|<^|3%nD`_nI$0zCUIpM02BVf+h_kl>U;vH-ssp1~2*A+M13OMtmKKPvMSB{UAg`#;)*YKEc(fJEMV96ZbU17g`5xK*jL&{>wMb=PwRlzIkyB*f8H|mKcB>ZhaZuQ`mm~|7>%IYwOPeL-e>;U1RUA{?HU|*u#=`X?_uV<W()%mQ4w5i&rRZN$x~Ipa3Lp2Wb7I%S!sLGYB+^rn+SbVh>6cuKlmI==7dr!N$k73=C5yGHAkQcUt3CyFchpZ^8(C|C8Wx!vlc71jnFg>zrMYDg{Sxe@Plb+pG~ZQN$_-LMVH6i-){qys}hpj`!ANpO=?7+;gXvT@8QuSncex_$AUM-+%QUD}H>^J*hTM(;lu`7o$lD(V0ffZrr+%wq}b;g56vD%x!%6%x!#G<FnCP!|3<}<j==)3fI=s>eCje_gZtUd@<?|!)|#Jj|Z(z9IoxHl~6K$vHm>!=6~S@O^n`w$bm1-E)~a)Ub|}bhikS^K8z<?09PQeHfjp16M74WONqVS8n78i!o1$9ZZ)rK)vZNy-0DxN-A_Bs&oqTE#MFMqN_<)KacgHhZYHDIq(gW>NkXOot+6dZQ}Mr={0Yh4$(j-fOxeM%Yo~q<P7K=QRy%w%Lp9fb-k@5&we_=j2>iB(k?yHGe^bU?;t$DH(@cYK0@kBH?2R&WD){d~cyfxJZVhk3<wxFwkVwwLNiQw(nQ_>g<MhltwP&XKW$iTj4&yky!$1!AK5r}^L+}8#GRsbMqCHCDy+iiVw4b6SXRhpBKA^9+&tFL=6{({nz_0f0l+)-V$E^-=<7Yywl1^(7Bj?rCw8=36!UqYhizQj_iHpJ*#l4qH2aw-DTUu(=cMbQD387Jm7bF3K!pfquQ!ZB)+b?wQ#Mh90GHp6vpjd<=(w4h1(_kKmrFPh{q=zk|k)_BDN~fe(c6C|h>wY{B1%W{$wiCG&0upK%B4SQ}u|*O-=V`!0jLJASQ3iSHPVmmUTZ2_G<er^iyG>e(FkQA69A2S6q*W6a{_dwbJ3}tOr(he_M`II~1c@dYtHBo2PMaDOw!*9d$x2uR$s$&FMv9NKar&*6nFYR6AH=tC0D%~~BI-I0!>tlxg6x!_BrKpC!sImDY@IcNnx(phEJov8Nf=_#_)7Gxgud9<Fm&p#2gnRr#s4=V6g`cC`-mS(8-s~G2{vv6a7A(el(5e6?5d@J8sqNl&}7TS=_nu$b{co#WLoA^Nn6Cq&;bdZp)A%$E%Km_%F@m@-~P<O9o@~f8;+nE&0)XY9&pCFE_|!lkKS(oTLLlKXg-6?RwhAMV}=i~6IuVhGlHF=cu=iB4MJ_R6ugVd_qZ*61E@K9R|<YQZLsH<Bqx1QdtDW#l_R_(^m{2lk_D+HhPY)jCjJXPfX7qr54-V#@X&w+3TzEe!1QU^079dk%P{N&KOm|WLR}zdbA&YIpvB2~YaE9hk#QWen>@XDSCiVJO7^>5!VU?c@uZ2N;CLkAd^G8ncxcEhzy~Q^&Y$1$tnrnAsEz}cNNj`ugL?nR1oUxijA7_u+VL<+yBLPh4uZOdyrGz7Thr7(c@p3>oP_E5R`yKih^l5{z^y#1B?m*!f}D$!ceLz3J*u6GQ;){jml}=W)o6I8G8jj@Lq3Bbq=ON>(*zcLE6Kel-29*g;cGSxPA2WsllkdM=k(<4^kjMpBiT^wX>3etj^TAa8(yZX!+Z6PPyNypE(LKu-a*4t-I5p3P4?S@euvabjggA{uP*<Kz-v$F8%e3Oj&qKzg1p0};393LOyjzOnvIrbgDX-~o26ijfqgqFPe+R1S+cZ^N^@tX#dFgrZ>u@XxD-mlp`OOZI&R~-)0=U;=QLRggyY>4Fp-DI9-nvAM~LPtk&jE8?_L}=fBfHHftLB8(VS1p<+2PJ;Z$bhQ?HcH;1-a1Cnr5WXLdE_qF`of4<4{fBp%oKsU}bU!V*r*9`FJ>3s85Dy|5Bt(#sxtBHjolsOP=9b;{8MS)XQ(s+?0gu;-`tJSIZmb2NPg>cZ(wQfyOrSh?9HN45BnI@SE+DQC0JseaOfD5sDrp1X^ftX@H6wqqDK;3lh?BPmX<52Y)D1{|0Ne4n1wPesm@b_D)+Y%wXr>&<CEp!DVLrN5I|n+TSZx-~jIxuiyj#$K*#%4}k45I6GlgnIw!Ku1IE)Bc`x>@aYZc*@@YR7S_5g>(!&h;-M7v$myC=28oRzL~u5=>IorY|Nb@X|ks?+J)*C?`MrS2WPw<xm#D`-K+9fAeYG7o3zz}v*-7*QQO{9CX>$x&o4g?j6lXx%Z{TiTujk8GVRnWkGDLqR6)Hp4NvH>B|v8VtG!yZ`k-MqZ#ij18mhAOtRt6*9kR=`>9Wz1Jx5e*_*~bo=D2~PA(A8qdIPa>G9x-9r#-kV-0jsHy_)AYusyh3wm}7}%%$(R#MS5y*pNrs9CF_80!H9f|1x$<2JM-12V=}uV5oAbw7)-0kcyuD|7m+JgJc?ygV3;&V%1_UZ?q#u^kI%bMw&AiYrj=zU&k#d^mabs3KJ%!EVD3QdPd}KuL29mvxhtQUHxZp3<g??@G-UgiRTgD!Z&O)R8^UK*!c?9;ge~4CHJuL4cx=04D%3T+!f;e!m9p-bfeD*rTq<D#i!E$OrApWEst+3n#%DNFic?l623cLm-8<fAc*djkb5r>tqzr=Q`-nmT#80qF^91v!a_962Ad5&k#I>b9@m(1m4grImwRH+Jxl~Lw(6hCW%}pg)`p3uhK6@dD(<O)GR}D(j}cJq0sVeRyIU@k#N9AN^9-RG>pBei#Y{ScQ`xzbx-k00$O}wU6=E3S>S3C<t<$FmFnzfw72y$+QP|L8DlPpo6LtY&EMX!gCKt(Dx`fBiY9u<`kbsdQfH{C~6~41v8J-M=SCmjvTCaGaBtK}6q-i+(4O0Z+$rI6CGz9=}G&`qCogE9}Y3nZ1s2Xl=gPDaQ3Q@w?N*oU{ZQQPYXu3f-J=r~U7&+ag^ju^jrs~R~M7K@`^*%{B6Wj$N=K_v^bXGhm%jDD;2X6ZFTG|U5|D^3pcj&^1RGoG@nH8)N^}17iT#J4KD_=Z+=s#!!R0F~~NbnVa=P`3E!B08<K3!{}FJ{G~uUTuMJ?V-UivyO@2<h5}YHB4rCyw3G@<l)6-90yEsQ|#G%t<@tPBw1gPH-R6RIM@zYYRu$OHdCrys-%(*f&>jX*@lR=@_dA2yn1uYh)8KZmLkJEBQJ#J14Ss!j7a(m;<gQ?B0FVDkA`iG-pwaUZ&AN-P=k=>1&+#CwBwt5JX)m{(89o>ea#iv!6_{d)Wf!y+eOS!!WN3Zw+tkt+6$qLF8P!AeJ6Qibd5fMi8GhJ>>hE0!v|Nx~IezoUF~)Qo%@v3N`hTMd&)W8{UeP0;GTUpK9sj>zB{%+C|^#hMq4R=#<RHyqe=lj7+<s-lwOF`Ys0buyF!H;zAx7*e1}D8+X2xFX@&1B>ps`74QEVbDY-c$euZ&PO`QdQIf+{l@M8A)Il0wYbYUBt#Tbt<+%;rOOuhd*&A|1%5lGQSq`2dI763H#_sZo9GRR`NYn$K+qs^H&zMe$HnF-v+bFD{DC$N;DTdIgzxw&v{;THk`~7$COZxEQ<J%YS-plEHnQun8&*~|YO2$=ZG_Qxmo`~9zN(HR61!<=*yO#O;(QB;cwv{5jE`n~NC0C375-Rf_Up{~S;tj9ckLj~p%r-tAB_s=sCRa_IU*?ffL-3WcP(*K+@puZuFHwVs@Ar>?DqnRQ!NDOSl<ARR^L)IC3Ol0-H_wuMKimz8k8He_PJ3lp8j>jd?V#={BS9;&-{7Sd-Vd$nnu{GQ+qyC2?J$eUEMp>^6d{7NopoU*VHk;2ztq4%BVBehz-*eR?I)=wz9sDbOKkj(6*K6MEX!H8U_lw4MOL*BDSOgBX_VMh1ge^|6Z|R#4I(Hn*|c_fi3s9}y$G<%f6OX7pda%Cn(q@;p#RSg=r1aqAe)W-zZIL}RNMjV3`THt9!_2L{eHPMMm*NgRl)nP!&y~%4Wjg3Fv(O@C8l@0lHQWa^MlpB)|`3=Zl?#gbXxQKG<E+Hy!z#RBhAdhh;nIWma%9I`JMzU6ANK-uGGEB+aG3bGXgC{IcL*SfzD<592nYZVqCppB#(}%2rlS@TDkJz0gWwf$6y1mDQX*UGSZ`5LD7Hi4Djb4@<;NZ6bqe9PH~%hFGY285(ssP0N0NQC_RN(DNsn2C#NP~cuGwM+^?v{+eVvM3LGR!RF5{$;+$PWyb%2jaUUNNXHNid_?D)VZLrsxu8V$1CsA@<VRYFF5K*%=oOeZ4Le8*axDz?K9GYh^nzKJsjmVMsHG$EInMv_EBdYJ%G!b7zi=XnZ^Zlsgu(Jssa|r%qFlrdcKlNDLtZ9hT@PXg(f!A>DE8C@0=6?^uB{WY@KNM@_O5t(8+xSnMdw~`%9vL0t7Y@X){c$KLG&=R>d53C@aii0feLJe+uj>5vdZu^Bl4fl%qC)}nzD65CKk47pXNR7f9Gpd;L!cw4GVD*OOtJ0Scmi9CDIH#}R=%1IrhUBa#7Q!O`fEmgabsVM%#pJHj4K@1y;SZoj(tF`SBO%@$|zL@S<6v-Eek1itDat|q~suF_LVHSt7-LM1=MU6nOaFXPor<>zLx7fN3ZVGIgE}7DMM6j9ux2T;+XlW0-L1#JYAqI7>WDYAps|ABuB2Xaf-@Nk_U~=SF!CpBr!3OYF8un1w0W6D<eERO-2BA_eILRh&*;@39RATEQ$M_*Ap8xdIF%@*%~2Gj%<N&UM!e<2msTyr@#o6{=_1GVyKZ0S&W~hhy#p}>(3{nAv+v-x1wk`)l6gn+`W=o??7Q-<u~xpVt4!O$6wz3)O`Q*PcPmaOANcD9K2#tkfRr~$yNKtN{O&M)&K&qs8G|&$i%{DjTAw3>MZz(M5IRu>e$kW!jPDmiv~w&mXO%S<xo5uC2QKl0uMuOAxx5LiIWP6-)Itm{SmoC?sG|c7LiuRIc0M>+_bg2UE;TmfEHvt5oLfR)Ahy}A9Wg{8*HLD@p7jN72hIBo|f_aO|%u7G8mJF@$@UKYSYow=u_&gXNw8L`Tt6E5L_H+Jbw)nD7(tWni3eW6?90J8v2<N_?Yb4EwsY@I!7H*i^Bd?k9j+u&ST<2bnz131WHfdiF>Q#IE;#~=xRRLCz@_U>TRi>geo&jf;#rX_8L~P^Nvl2jwh$*yeYl-O&dnTYy0$jd0(VD4+dPTE(zfzSd!gxR6MgG%O9^tFQB3!*)}+Y#Hi+lYFdhe0L4mhK7vm^A&f+Fbwc+e<ia3<`Dk(p(1v1$W7BZT`;0=aZE}S&fC7!Gkrd7ZM>c1q>0*y{sgMk4!4j*@5g#+m>}0LOj%RCv+ZXorIr4)Hn>=r*7jo7F@Bm-<b}6dIpw>^;IGlTvp#a2b&){2PiB3gRm6qb^c%{HDNt@!gE@LuI-87txM3f&M5t;(GagU6cgdGx`(|U}Hr#LMyp9#1MB9a2)1OcH1#DqatVtvVh03DlL9xi(B4CsL9V5W0_4ochIc_`o%%tU-)r4;9{48zwYgNkkek%f0%3x%^TF?h+4i63$NkzF#EqVq0s#(92AklK8mQbojr-4og<MAfaz^tHE;W;sHv#~k=A*Q9CX3TTL{nE|gQ_ZGQuM78XYZ~($PGZGD9oQRis1m&|cA&0$5r-PTbt&__a`OB2VE2`!1IK0~68m=lSb6Oz=tmI(=XQi&Hnz@{cT){xkE9Z`(oVxKDhPq@qClzDp&En*$(zrq*DLVP~_jGQ5BJG6!E;NaDkVDDC0rrWdp;M31qDH@?W`8MV==NJ@Lqg#FPC}i%0A`o?VXQ2G)3GS?MNRPh<^B;6Iq*fOR|M>|!(ThBai?aw{Yl)J4sOc9OJ?Z^OsTaotmO&RM+45Vg>#Q&7DoO`X6>tf0`IxMCPJ{NC6KI=rhh_9hQ5gTeM)+mS53q^AKWmKy0&B3-(XtiG9&scCKInS>NA%K$UjxSXa4(3u;1;nO)|VA*eSrX_9t$xHAePi=6znp1Xl6z@`_Mq2wXYgYoS`DwndD=UxGm#S1Z*=B+U+m<`PS}xwRQ@Y;SZo>YLk}oz32Mx3{xZYgaaVJMDOTtNOUIz0<DrO2PgZ4v*`#a-~d7y;quv^y1E^0JLvNP~2*du4Bx7KqBt0CL94DcxF<-o=ciZ^(Ns(?S87$h7h(qhXVWm2NvnxrT"
INCUMBENT_SHA256 = "331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50"

CASES = (
    (1, "decode 1", "normal", 2826763372427500438),
    (8, "verify 1", "normal", 2827019192515550467),
    (16, "decode 16", "normal", 2826848645790183780),
    (32, "decode 32", "normal", 2826933919152867123),
    (128, "verify 16", "normal", 2827104465878233812),
    (256, "verify 32", "normal", 2827189739240917158),
)
VARIANTS = ("BASE", "NODQ", "NOB", "NOFIN")
WEIGHT_BYTES = 3 * 4096 * 4096 * 9 // 16  # codes 1/2 + scales 1/16
BUDGET_S = 240.0
T0 = 0.0
_RAN = False
_LOG_BYTES = 0
_LOG_LIMIT = False


def log(text):
    global _LOG_BYTES, _LOG_LIMIT
    size = len((text + "\n").encode())
    if _LOG_BYTES + size > 9400 and not text.startswith("MD_DONE"):
        if not _LOG_LIMIT:
            print("MD_ERR log_budget_reached", flush=True)
            _LOG_LIMIT = True
        return
    _LOG_BYTES += size
    print(text, flush=True)


def error(where, exc):
    # Bound failure output too: compiler exceptions can contain megabytes.
    log(f"MD_ERR {where} {type(exc).__name__}: {str(exc)[-140:]!r}")


def over():
    return time.monotonic() - T0 >= BUDGET_S


def incumbent_namespace():
    raw = zlib.decompress(base64.b85decode(INCUMBENT_B85))
    assert hashlib.sha256(raw).hexdigest() == INCUMBENT_SHA256
    ns = {"__name__": "_moediag_incumbent"}
    exec(compile(raw, "<embedded moe_verified.py>", "exec"), ns)
    return ns


def diagnostic_source(src):
    """Auditable changes; every macro site is listed in this function.

    NODQ: dq4; bmul2/bmul2l/bmul2h; SCD/SCDK; wgs_seg's sgp.
    NOB: WS B range in moe_fused; WGS B range in wgs_ranges;
         run_phase<PB> early return (WS consumers). Producers see zero B work.
    NOFIN: moe_fused's FINPF prefetch conditional; final block through DBG(5).
    No barrier, counter reset, A epilogue, launch geometry or weight copy is
    removed. Both helper and consumer ranges agree in NOB (no dangling TMA).
    """
    def replace(old, new, count=1):
        nonlocal src
        assert src.count(old) == count, (old[:90], src.count(old), count)
        src = src.replace(old, new)

    defaults = "\n".join(f"#ifndef OKC_DIAG_{v}\n#define OKC_DIAG_{v} 0\n#endif"
                         for v in VARIANTS[1:]) + "\n"
    src = defaults + src
    for name in ("bmul2", "bmul2l", "bmul2h"):
        head = f"__device__ __forceinline__ uint32_t {name}(uint32_t a, uint32_t b) {{"
        start = src.index(head)
        end = src.index("\n}", start)
        old = src[start:end + 2]
        body = old[len(head):-2]
        replace(old, head + '\n#if OKC_DIAG_NODQ\n  uint32_t r;\n'
                '  asm volatile("xor.b32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));\n'
                '  return r;  // all packed-code/scale word bits are live\n#else'
                + body + '\n#endif\n}')
    head = "__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {"
    start = src.index(head)
    end = src.index("\n}", start)
    old = src[start:end + 2]
    replace(old, head + "\n#if OKC_DIAG_NODQ\n  r0 = r1 = r2 = r3 = q;\n#else"
            + old[len(head):-2] + "\n#endif\n}")
    for macro in ("SCD", "SCDK"):
        start = src.index(f"#define {macro}(")
        end = src.index("\n", start)
        old = src[start:end]
        args = "word, b" if macro == "SCD" else "word, b, k16"
        replace(old, f"#if OKC_DIAG_NODQ\n#define {macro}({args}) (word)\n#else\n{old}\n#endif")
    old = "          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;"
    replace(old, "#if OKC_DIAG_NODQ\n          const uint32_t sgp = sw;\n#else\n" + old + "\n#endif")
    old = "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);"
    replace(old, "#if OKC_DIAG_NOB\n    S.wsr[2] = S.wsr[3] = 0;\n#else\n" + old + "\n#endif")
    old = "  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);"
    replace(old, "#if OKC_DIAG_NOB\n  S.wgr[2] = S.wgr[3] = 0;\n#else\n" + old + "\n#endif")
    old = "__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {"
    replace(old, old + "\n#if OKC_DIAG_NOB\n  if constexpr (PB) return;\n#endif")
    old = "#if OKC_FINPF && OKC_FINCOL && !OKC_WGS"
    replace(old, old + " && !OKC_DIAG_NOFIN")
    old = "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]"
    replace(old, "#if !OKC_DIAG_NOFIN\n" + old)
    old = "  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered"
    replace(old, "#endif  // !OKC_DIAG_NOFIN\n" + old)
    return src


CPP_EXTRA = r"""
std::vector<int64_t> okc_diag_info();
void okc_diag_stream(torch::Tensor buf, torch::Tensor out, int64_t grid);
void okc_diag_empty(int64_t grid);
"""

CUDA_EXTRA = r"""
// Contiguous streaming floor from bwdiag section A, explicit noncoherent v4.
__global__ void md_stream_k(const uint4* p, size_t n, unsigned* out) {
  const size_t per = (n + gridDim.x - 1) / gridDim.x;
  const size_t s = per * blockIdx.x, e = s + per < n ? s + per : n;
  unsigned acc = 0;
  for (size_t base = s; base < e; base += blockDim.x * 8) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      size_t i = base + (size_t)j * blockDim.x + threadIdx.x;
      v[j] = make_uint4(0, 0, 0, 0);
      if (i < e) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
          : "=r"(v[j].x), "=r"(v[j].y), "=r"(v[j].z), "=r"(v[j].w) : "l"(p + i));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) atomicXor(out, acc);
}
__global__ void md_empty_k() { asm volatile(""); }

void okc_diag_stream(torch::Tensor buf, torch::Tensor out, int64_t grid) {
  md_stream_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)buf.data_ptr(), (size_t)buf.numel() / 16, (unsigned*)out.data_ptr());
}
void okc_diag_empty(int64_t grid) {
  md_empty_k<<<grid, okc::NTL, 0, at::cuda::getCurrentCUDAStream()>>>();
}
std::vector<int64_t> okc_diag_info() {
  cudaFuncAttributes a{}, b{};
  TORCH_CHECK(cudaFuncGetAttributes(&a, okc::moe_fused<false>) == cudaSuccess, "attributes small");
  TORCH_CHECK(cudaFuncGetAttributes(&b, okc::moe_fused<true>) == cudaSuccess, "attributes big");
  return {a.numRegs, b.numRegs, (int64_t)a.sharedSizeBytes, (int64_t)b.sharedSizeBytes,
          (int64_t)okc_smem_bytes(), (int64_t)a.localSizeBytes, (int64_t)b.localSizeBytes, okc::NTL};
}
"""


def build(ns, src, kind, variant):
    from torch.utils.cpp_extension import load_inline
    defs = tuple(ns[kind + "_DEFINES"]) + ("-DOKC_PRMTSC=1", "-DOKC_DQ7=1")
    if variant != "BASE":
        defs += (f"-DOKC_DIAG_{variant}=1",)
    tag = hashlib.sha256((src + CUDA_EXTRA + CPP_EXTRA + repr(defs)).encode()).hexdigest()[:16]
    return load_inline(name="moediag_" + tag, cpp_sources=ns["CPP_SRC"] + CPP_EXTRA,
                       cuda_sources=src + CUDA_EXTRA,
                       functions=["okc_run", "okc_set_pdl", "okc_diag_info", "okc_diag_stream", "okc_diag_empty"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                          "-DOKC_NS=3", "-DOKC_NSF=3", *defs], verbose=False)


def generate_weights(dev):
    """reference.generate_weights + nvfp4.random_weights, only local experts.

    Same distributions/operations, but smaller leading dimension changes the
    generator offsets after w13: not bit-identical to slicing a 256-expert set.
    This does not change routing (it has an independent seeded generator).
    """
    gen = torch.Generator(device=dev).manual_seed(5252)
    def random_weights(rows):
        q = torch.randint(0, 256, (32, rows, 2048), dtype=torch.uint8, generator=gen, device=dev)
        z = torch.randn((32, rows, 256), generator=gen, device=dev)
        sf = torch.exp2(7.0 + 0.4 * z).clamp(8.0, 448.0).to(torch.float8_e4m3fn)
        mean_sq = 4.0 ** 7.0 * math.exp(2 * (math.log(2) * 0.4) ** 2)
        gs = torch.tensor(1.0 / (8.6 * mean_sq * 4096) ** 0.5, dtype=torch.float32, device=dev)
        return q, sf, gs.expand(32).contiguous()
    a, sa, ga = random_weights(8192)
    b, sb, gb = random_weights(4096)
    return dict(w13=a, w13_sf=sa, w13_gs=ga, w2=b, w2_sf=sb, w2_gs=gb,
                post_norm_w=(0.1 * torch.randn((4096,), generator=gen, device=dev)).to(torch.bfloat16))


def generate_input(bs, dist, seed, dev):
    # Reproduction of reference.py:generate_input, including RNG draw order.
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=dev)
        u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    elif dist == "skewed":
        heat = torch.rand((256,), generator=gen, device=dev).pow(6) * 4
        scores = scores + heat
        idx = torch.randperm(4096, generator=gen, device=dev)[:40]
        x = x.clone()
        x[..., idx] = (x[..., idx].float() * 30.0).to(x.dtype)
    else:
        raise ValueError(dist)
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256,
                       "local_start": 0, "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def bench(fn, flush, n=24):
    """bwdiag bench, returning median AND min; events if graph capture fails."""
    assert n >= 20
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            held_output = fn()  # keep the allocated output alive until replay ends
        torch.cuda.synchronize()
        run = g.replay
    except Exception as exc:
        error("graph_fallback_events", exc)
        g = None
        run = fn
    run()
    torch.cuda.synchronize()
    ts = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(n):
        flush.zero_()
        s.record()
        run()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return statistics.median(ts), min(ts)


def measure(bs, variant, fn, flush, touched, nbytes):
    if over():
        log(f"MD_ERR bs={bs} var={variant} budget_skip")
        return
    try:
        med, low = bench(fn, flush)
        log(f"MD bs={bs} var={variant} med_us={med:.3f} min_us={low:.3f} "
            f"touched={touched} MB={nbytes / 1e6:.6f} TBps={nbytes / max(med, 1e-9) / 1e6:.3f}")
    except Exception as exc:
        error(f"bs={bs} var={variant}", exc)


def run_all():
    dev = torch.device("cuda")
    if torch.cuda.get_device_capability(dev) != (9, 0):
        raise RuntimeError("this frozen H200 diagnostic needs sm_90a")
    ns = incumbent_namespace()
    src = diagnostic_source(ns["CUDA_SRC"])
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ["MAX_JOBS"] = "1"  # four independent extension workers, not 4x ninja fan-out
    log(f"MD_INFO device={torch.cuda.get_device_name(dev)!r} reps=24 flush_MiB=256 "
        f"raw_events=1 seed_source=f0f01b3229f94cbb parent={INCUMBENT_SHA256[:12]}")
    log("MD_NOTE NODQ=wrong_math NOB=zero_y_finalizer NOFIN=barrier_reset_kept "
        "STREAM=contiguous_unique_weight_bytes EMPTY_MAIN=trivial_same_grid_block "
        "TBps=effective_not_physical MB=1e6")
    ready = queue.Queue()
    stop = threading.Event()
    def worker(kind):
        for variant in VARIANTS:
            if stop.is_set() or over():
                break
            started = time.monotonic()
            try:
                ext = build(ns, src, kind, variant)
                ready.put((kind, variant, ext, None, time.monotonic() - started))
            except Exception as exc:
                ready.put((kind, variant, None, exc, time.monotonic() - started))
        ready.put((kind, "DONE", None, None, 0))
    # Daemon workers never hold prepare open after the budget; each starts no
    # new compiler after stop. An already-running compiler cannot be cancelled.
    workers = [threading.Thread(target=worker, args=(kind,), daemon=True)
               for kind in ("TINY", "TINY2", "MID", "MAIN")]
    for th in workers:
        th.start()
    try:
        raw_weights = generate_weights(dev)
        if over():
            raise TimeoutError("weight generation budget")
        # Only the private embedded namespace is modified. Its original prepare
        # performs the exact repacking/scratch setup; compiles run above instead.
        ns["_ext"] = lambda defines=None: None
        state = ns["prepare"](raw_weights)
        assert state["sfx"] == ("-DOKC_PRMTSC=1",), "generated scales must take incumbent PRMT builds"
        del raw_weights
        if over():
            raise TimeoutError("incumbent packing budget")
        flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        flush.zero_()
        stream_buf = torch.zeros(32 * WEIGHT_BYTES, dtype=torch.uint8, device=dev)
        sink = torch.zeros(1, dtype=torch.int32, device=dev)
        cases = {}
        for bs, load, dist, seed in CASES:
            data = generate_input(bs, dist, seed, dev)
            ids = data["topk_ids"]
            local = ids[(ids >= 0) & (ids < 32)]
            touched = int(torch.unique(local).numel())
            counts = torch.bincount(local.long(), minlength=32)
            # Count re-streamed expert jobs too, so unique bytes are not
            # misinterpreted as the actual kernel's byte traffic at M>=16.
            jobs = int(((counts + (7 if bs <= 16 else 63)) // (8 if bs <= 16 else 64)).sum().item())
            kind = "TINY" if bs <= 4 else "TINY2" if bs <= 8 else "MID" if bs <= 16 else "MAIN"
            cases[bs] = (kind, data, touched, touched * WEIGHT_BYTES)
            log(f"MD_CASE bs={bs} build={kind} load={load.replace(' ', '_')} weight=60 dist={dist} "
                f"seed={seed} routes={local.numel()} jobs={jobs} touched={touched}")
        log(f"MD_SETUP seconds={time.monotonic() - T0:.1f}")
        done = 0
        seen = set()
        while done < len(workers) and not over():
            try:
                kind, variant, ext, exc, secs = ready.get(timeout=max(0.001, BUDGET_S - (time.monotonic() - T0)))
            except queue.Empty:
                break
            if variant == "DONE":
                done += 1
                continue
            seen.add((kind, variant))
            if exc is not None:
                error(f"build={kind} var={variant}", exc)
                continue
            try:
                ext.okc_set_pdl(0)
                info = ext.okc_diag_info()
                log(f"MD_RES build={kind} var={variant} regs={info[0]}/{info[1]} "
                    f"static_smem={info[2]}/{info[3]} dynamic_smem={info[4]} "
                    f"local_B={info[5]}/{info[6]} threads={info[7]} compile_s={secs:.1f}")
            except Exception as exc:
                error(f"resources={kind}/{variant}", exc)
            # Run the exact incumbent kernel_fn, including build selection,
            # allocation, slicing, and the okc_run call. Assert its selection.
            call_ns = dict(ns)
            expected = tuple(ns[kind + "_DEFINES"]) + state["sfx"]
            def selected(defines=None, expected=expected, ext=ext):
                assert tuple(defines) == expected, (defines, expected)
                return ext
            call_ns["_ext"] = selected
            call = types.FunctionType(ns["kernel_fn"].__code__, call_ns)
            for bs, (case_kind, data, touched, nbytes) in cases.items():
                if case_kind != kind:
                    continue
                try:
                    if over():
                        break
                    # Isolate stateful counters and NOB's missing B writes;
                    # no resets/initialization occur inside the timed call.
                    for tensor in state["scratch"]:
                        tensor.zero_()
                    measure(bs, variant, lambda: call(data, state), flush, touched, nbytes)
                    if variant == "BASE":
                        buf = stream_buf[:nbytes]
                        measure(bs, "STREAM", lambda: ext.okc_diag_stream(buf, sink, state["grid"] * 4),
                                flush, touched, nbytes)
                        if kind == "MAIN":
                            empty_fn = lambda: ext.okc_diag_empty(state["grid"])
                        else:
                            empty = dict(data, topk_ids=torch.full_like(data["topk_ids"], 32))
                            empty_fn = lambda: call(empty, state)
                        measure(bs, "EMPTY", empty_fn, flush, 0, 0)
                except Exception as exc:
                    error(f"bs={bs} var={variant} setup", exc)
        for kind in ("TINY", "TINY2", "MID", "MAIN"):
            for variant in VARIANTS:
                if (kind, variant) not in seen:
                    log(f"MD_ERR build={kind} var={variant} budget_not_ready")
    finally:
        stop.set()


def prepare(weights):
    global T0, _RAN
    if _RAN:
        return weights
    _RAN = True
    T0 = time.monotonic()
    try:
        run_all()
    except Exception as exc:
        error("prepare", exc)
    log(f"MD_DONE seconds={time.monotonic() - T0:.1f}")
    return weights


def kernel_fn(data, weights):
    """Prefill-shaped zeros: deliberately fails correctness, never submit."""
    t = data["config"]["tokens"]
    dev = data["stream"].device
    return (torch.zeros((t, 8192), dtype=torch.bfloat16, device=dev),
            torch.zeros((t, 8192), dtype=torch.float32, device=dev),
            torch.zeros((t, 8), dtype=torch.int32, device=dev))
