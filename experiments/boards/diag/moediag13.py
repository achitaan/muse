# okc: problem=muse_prefill_layer
"""DIAGNOSTIC ONLY: prepare runs experiments; kernel_fn returns zeros.
V12: moe_b8 BASE vs "STRIDE" slot = moe_b8 CUDA with OKC_L2H forced to 1 in TINY builds (TINY passes -DOKC_L2H=0); TINY2/MID unchanged (already 1), pinned TINY at R1,
TINY2 at R8, MID at R16. Same judge-like input seeds and distributions as
the earlier moediag timelines. 40 cold-L2 reps, 256MiB flush outside events;
reverse case and variant order on pass2. Saved BASE output for byte equality,
max absolute/relative and relative-L2 errors. No GPU claim from offline tests.
STRIDE keeps A then B per CTA, with round-robin global skinny units and
per-job A publication. No promise that this alone overlaps B with later A.
Separate BASE_TL/STRIDE_TL TUs stamp entry, Wready, Adone, Bfirst, Bdone, exit.
Bfirst is phase attempt BEFORE readiness wait. Per-CTA earliest entry/Wready/
Bfirst and latest Adone/Bdone/exit across consumer+helper warps; report
min/median/max over CTAs relative to first entry. TL perturbs code/timing.
12 small-kernel CUDA TUs compiled up front, 235s cooperative budget.
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
_FS = []

INCUMBENT_B85 = 'c-riJeOntxvhV+VidputGh>9TkuWC6I2UYVb8KT@3m@-3EGv(skr0THn2`{~$@%QJepTH)Jul+LcCvfVI!_WbZ`Iw^)z$Uhe|5&E-A!i_k2|CAZ0l^6gzd9&*q*Ni{`%5io#$aZya;<vKOUbs|9uk9E-TLQbUH~kA3humr^mBSwL3n0c!4Jm3Dj!6R>ybmhfzeI#o@&;8IB{T-*^<P`C;gH{YKdE*Ly*y(|`Q9zR{@l)*g52jn3MmM&sv3)7hQ`-Q#ezUaeIL&YN*J7&+bJAR2_pCc!%!UxZF?7!0CuG97l4Cf!S><2ZEQ2JvVZIXk=ly7ThI+qVw~qj4t~ttJE^Bv764Eb0aEFie~v>dAXyzaMs|G(smD$7jww91f1B6#_X8hS4w@IJ@r~b$Lc#PLIz*Y7_lx9S1RuFX_g?w0le+oX>{T5htM^VKg05r{i!k3a+Zo?rb!2M#F9xC85(Fk4EEp0uVR;BbadB$fwtT@!w(0cqX(}oc{`<SrA`2^;&K1n$IwgG>!B5yc%#*)p0y{sCt?_q%Y;;gBKs(ww&$#=gzZt`_K0d_TKHcoZWXHoR6&+73af?_aELp|M(2IEBw~;z1G2py`7J^V}SToqNP3|bxPgj@tdA4s}U|c$#FoqItzoynbOFoVSFZ&)g4DY`I0agJ4B_JB%<2htc%+f0}OBXhRHM@c4m0u1c}o_!!%o+E2kB91&mKE#N*lE*!kHR_vvCt?~S{&Gn&1!-i_lz_qyZBRSbf0#`7qQ9qJgSetP8uv+40T9{$R08c1c|Opk-9Lro9jfIcOHo2E0K>qa;rqI$s%6uL5tFjO8#=mcGcr29e$p?3|SF}<QXilx9EAoNb-@u)&X9_ot`cfNvA;V!Km`i$24**G$=)T_>%mWDu*Mv3UhG52LMizj2ErWD792+}!QR&bViK#AiHJ^6S%5910=XH2t(jtnFDU2&#kryCF%;5`FJ9x^6k=PZbVL5S%@+sTYpsdTR5%#TAx0?`Av95D1OhUP=i7{PIe)El0x<ajs%tp2c1c%6iC7r@;|wR;{n2qq*`kb;^`lPS#r<|6^iNHk0W+6f~<Mi;yy1K;XvmIVGjo-I2reUD%9vS(8P{R`||40|)ciyfO@42bZOdQBXex<max8zu=a2Btn)AUsK>1lXcByR=CA;D$N+o5W#1jKLB4X`k_bie}Eny&(-U;8mAS`7r8^W{ehEro>>U#39dy=mX7cGVV|3Ai9K`q)F=0T<F!wpbQ{+rJ}h+e>j-M{02?aD9kaMcbyX=km3-6=t}O<1kFagdU(S-3$gQ1@6+W&)Dk6lH_(K}cSh>6?*xt@iUC%#12s@`1Eys)8DhbWxr1uJ1ELQa3*F8PJ<IE8%ooD@6F`#HeHPLV=}fOCc?5hK$EOA6F{hi{bLLv0!gPtyhN^~REf7JY8sJ&bBbIX!3`YTp5Q<M&T&w~P4ASfdio^noe`sz<2%*;vI~1(Z>qACyFr8w5VqEDS8z3$XBfJdGCe%88N}B>L0{Im0Zcip5wSGy<jdoryiw_oW0u}&(nq)ao6=<m(8wv{smS95<pc5pR&WM+5&k~}9v!OAbKCG1a@vwWGVH-4kB+`fwAp%qa3MV9~b>xJE?QyKH2#A_^d$k#G*eSz==!fSbpa#byUNZFIa4?Lhogxhsd3egz`WZG+aq?(XSOk@;ndAo4ZbL#GNmuC#4ktm(v;+L`V9vrg99_{0iB1`T9ikslP86Jlo+g4Jv8=csba`{BWOiAD-a^NKrEuKOQo%E90!m&erGWhEFpE0VszIh&ka|Bhy#ct)3^8r@lw^~_8VjEIJ9=Z$AS%{!nPR^fQ{$tQ*(b9OaTvv7v=qY>$NlFnszt(0G50TsDCw;8hI#EiV+%gu<J<wPf=)<C?Gr+-6NYXlH8{&=T+0TU$PI>#(`Vs`7H2#r=3T*jkRrg8HIMOS#DdH$QuE+oZDi3OrU(PRrb)VpcqWp{wOg3Q;p}$k-?U%EIfo;BJ|c0I;8~lU5v-D{WE!3&S(Z<`_AJD<)8!pjy(7ecT~j0<BfZ(^uVmOvh9)e$1Ain#cV`Jp3f$%y^Hn9Feqyd@Z)oAAMn-nTnnGzn$z<4_jb{lh?6V*~1$&HBSt}Ld!elszc&{Mh#C$P|OKB7AY<ZtBA2{|Jt}d4^@0<*8hOgJ=Ed(eVnc$;mxu%_C>QW~p3L{Z8WUfzLw_CE8r_vTjXu3Nl+{<<kaoSp#DYL3qotF^R(D*Zh3N5WUt(k1hih@hkr_2(xd3Z=WtYe|d0h6J=vuwz+1qq!rT9O+l;goQc5+$+7QExtk%o2^GRi=f6#*WKXlAz)N)V1+dFq&Si_T!MQh9oRrkZ#`@g(cE})|RPJt>w`-wEiZb?*c!x+5cqLp|1&}M8MF$E9o8Txopx2-%-MetuJQP3B$KpATP99A`CG1NvEq}P1pMXEc$=VT)FgNIGJJrl7chU;?@ur5LUrFXCfoE=8{wb0(Tr-ge*0tonn1{-0wqjq-`S{5wn*66W<xfQ<)~iE0hqf<V<Ep>G*&OnK9jRFqw>?>W(8K0>(FZoa$b86buRT@@{5`gii)z<Js`oh*o!!1o4p9Za*d-rS+$9nC_W2Yj6{f6!CEss-2V=CCN2LZ{QE}AEd()4O{KhGzbwIvmI28%rPl?kE}X-eNcm?c_qZrK<j3Lro(}BBNzmD%*;@k58QN{v*h-8oFuD^It->eo<SCqYodd|83prXHk@L>qi`VG24PMQz?La;{PQ|S$h(MiK2gfP4XK+>)Ri8bo|`k4ehH8Wby?a>MK7%gs}-b`nRM-`tx`*8vYTM>VkT4C2@);*_2{-HM}udA2ko?%vWm5;^C7hDR#oo6S#Xtdz&x)aemT@KQjTj~BQ^2_K@g?pX2cAcvLG4J-?6o$XBBr50v2$*N~&}*YHV^l3qzTAv3!bzpgG*8y-&M=Cu5!w1rE?}bW`*in(8497R=2Ssx6%q{}hHC@Sc*_SDI=YPFwC+EsI$x71g1)!IB^*rx>I~YRWJI{fO3@WI6$MXb2uq#dzHlBmh#e{jSx%*q$p|GPjHgtc6g&^25wDZ<<XBrGizM?yeO@{S~m$UI<yNk`YXpfT!t_RHGD=NJ*cHOgJkM3cuA*xP!eh3(T}XVX#20tjc(5Z+co!$Pc)Pz@7&-XXrvHogjea{dWg@&t5FkBE6h4{;@b!OAzcct=TKkVizTAseomky$mp|`UW(oL60@ulpMoS6oZ8YuvKRODK^MlM25oysoVy~89?RuF2ykuAVC?0K>`&n>wi!mrE8i*tu7ois;hzSXNvU{Q#ndYuI)$D-UKp)mtFf=6PYvYr#uOp?jYSki!F@fN*TTbEkjzCIHiY_V5?tPLoAAr?kN*TAmiRDhI(b@363SG>=04HfE3_^W6}M<>B<mp%|44EqRbm}B<U-u>I`Wo-zBx~n6q7Fd<|y133@&Jj-g@MG@5~<2cR%@oAptVc}QqVvLj<H4#t*VFN}J#Gc6KlC`hv&(LD9k7C9fITANe|9i@wmEor2si98X{iWDgby3QX{imp_HVWF6v4<btBOmmru0T?wRyX!C<4j~g~r05bo!8SpbxW)vK<q;eAOL|nX7gV2B)vE=Hk8MLVi;N+n@r>0m?L@&Zm)XaSWDiR?3+te)oUzF0MLCV5Fg=f0y;9x{(UnsPGpRZsBie-%p2F~wn(Gdsf-_Xh`D58!uJSUfWp1;~Yzs|mwoZXo^Cl({BRjVB{B8=N65hD076}Ou5siDP=T-jMA5Za_@lWxd)ESE^h-E!swGtZ-cQ2XI_K}3WP@G*@AejkLjfspRMqR?EQNjkKOY(xaQY(k`!0<BcX0`-o(*`kdI0$0#p5-+i%d@dgj8ls;37DuQulL5xXQv|CTV5|rf^r~=Ae1_1f-~6njC2IsTo_-#KcOzEXNtDvWlhYw-xb>~rqpB=A|O;k0zpDE1d)KJm_&#18C>@05n-HUmo5!dO^c=Ez`|UxE9;e}XHo4SC7VWyv&U8E`7mLX4X*3H^ND0z!qAmj9;U;cD^a1@0sz&{W)(cItT3h4x=K2Mia#W2hh6j-28^PTY`+haYi7DV*xqQHSl(_qd#z<>XS=o6N)i2O@8H$Dj|a}D?GGQe_Yd}7w48Sz?0|)LyUzCh-<{Wc`_D;m8j2VE5;on$9?y{ZPtP*%rmKtXyMg9sSEOh&%2)}D%NR1D^<eMd&5H`n!T##r{_cmp{g*G^zSuvgIB#Ekc=n2Vy1ldaX7AwdOjNsj2m3Et5?rxupuFGyKoj`!&GrZ9{l^dQ-?d)IZY>_t5j-U{iU~D5WG@PPoJ1$h6A;Z~Jf6fuC;&Y0K2aQAVwz0(vt{QM>u8dYbcI2u?0uN<UXYBtL!+$AHlV!BY)7`;%tduNrN@mb-P7pBhi`^KXE@?uf<5fr4vA9HlzSs^=+=l$wA545_%kLT?X4q{n%X9eC>)H2B+Ye0uVQ@vm5d?UvP#{!4qOp8;GY-`J1in{2M4f^Se_W&>=dyX2}cW*uN&dCnGFZVqI3dBLvCEzCwQuZvtW?*N8o!MD3OLtB$E)a1{ohbEj1FE#Wez9D2-lHU^PF7Wsv#~lY1=j1CRo&y&it^yrRu`of$qflRLvmO{~R%v(^ju+UNk?GYoz-79t*u$G!P*l(lo7(w07+OafRmAy>@M{eCbU&0^WZf>D1Kr9u*Ko~6+;@M(agZM;b1VM2ri%7g-tHy0Wx#wif=E{5zF?yCS7S{Vu%I`Bg^kgklMtIl>8y9n^1xis3}PB(gMv3xp)a*$ojMd#?XZbdEIc8|wnvD~t4HWzrnwqlY5`yuljVu{?#AnJxPn2DHf6_>nXS`5!3gy7g_J3;YCkH8ssM#@gc^3OxCCdky{M50k)9Vu;YnB<&Uq{zG)&!KgR0&5V*xV1p1qvYtHXq55s7>Qmvd)WM}ZiD5f443=N5>x6MviW~%!?FlN*|VXc4g12&!48BQFpl~bj(TCA6eIbDr0!mc`9Fxym{)1R-C)yR?6Wvdy@JZ_P5e8Iv9Ofsw^FoucdnGUVh`&Im`Jf}l!duPpOzqM`je9Ni~Z-=lgok&%{$xg-_vV*f7`^Qvekn4@0AMcu!AG<jJq_?_{$Kyc*~C}D##>j*GL8TF|De2G6IJeJ+hKo6#Bz()Jq)NR){6YM$v(HHYA!|K0I2sRU+6))Lwa|iHy0h(l;$_zUsKo$I(wFh9a|ybpwC#9JU&;nw*fBLi9n3gz1aYLadEH<D14>O0H-RxHL{+HZDjfh?~%-qXcd^d0Sa{Q_jzC3q2A2K^hiSj|JF?-psXQxf7<LJ?tG$dz0YPW$Gdu;K8+*vBzY+zA8)zT_utnTG0s#huZIKj7F)^C5XGn@Z6{5k-AV1ujrp6=aBnP{muEAkIW~Ebb6MimL;l+9g6F?c)`TCdQAbMH6Snz*#MP8K&uPGNGZb1Obwk`LX%~I7<bsF7i5f!n#6;ty%=r^`F%rr!hY)og0IbIx6uX`Ni!;rPXJj4z+#jbTq5=hi)H<96!f)#uaMXZ!%R2yB4A;OsEGy`4Q51)Bx2B}66GT&jU6Y|4U*#MnB<n1mp_boaMT-Ox{2fN|M%|uHRsdXw!iMY9l!9{Zxn@SL2q==K0e=ePO0T^R9#xy9u3AEhrYC`|5@7Q#Q@GVJ`G8dnN2VcpFbbGd*6Qj`Lm{o`lEHzFf@mN<KAUgVFDz+eY!;Cp3t{%-aXrX(|+-{_b)yiv<!?hF;KE-HG~mpz=HqVH}BD71$}dA$*(#Nh6-f4pFhVKH33nqh({#BF5V_&CsKQ#9c)8nL4ZDoLv&_Ca8&RI!y5?rsx%=ZmADx(4SxPilHO^1*h{FL)m8CGChSvA!l`m!1p;f#``zcyuEJL`8c#j?jIbSFF;GC5jAj_&$+*L}NCu140L3&VF0z<I79)bzB6{2Zd;9J7-`WT7UehWfh}-}qKU&P8%&d+Re$Wr54*)$`5)aVvEaFH80=2t+PzJ|qGWwA2pg=kglG9-nT?wEnL)s<oj4PgF^ql=yF?RM51h!-<m(`^@&C+|4Oo@XzE@skMA=;@>2arAC(^YFeeEH_%=g*iX5E#MzDbb_Cx@IIGV3wDZHjWFLeu7=Ui!dLda<uS}Xj5`55<+O&{TT^`42-CXmO?z9Ck|~$NSt^r?2p*UB%xJFG6a*=Yo5{1pU=*MD(zq0&!3w#Ta48u(cOj|cLb`>f+HRi5;^ru{C^^;9B1Jf{KiD2uUC_)h~O+h!0*6_fG#tE8GxGR@Q=@*IR=qt$nEs~b+5YAFgV?D+#W3^jm|n_kvR)!nP(`TDUW-~;Wg@(#8^+VfkIb>eMlI&X;c<tI2n${gISo)LxP+GeE<0~CXx2tX&a7)_8c8mYR6PBC6d1O?bMf3>^~R{coDp?k+-BpelT_@UXw;$kji}Ksx#{oWL_{#CcNH%LJR`C2b1#|0^bo8C%sCrNC`VL+CWz6W0<&m%t1y0>0`&*-GV5R>e3pabeF>_CnRBP()R@ZaC+oyp@BBw(#oG7LUcWH9yoDw9#7pQIiyh@IS-vzd(WT0*e67UlSGE{z(JS!Ik31*>iUp~mn7FAk^;i%9I6^43R9UV7758ncS9&YIHiY%fq~%h{_2iK6he2Ewr<`cCf-6Q^9-SXn|W@}J#ldp91$pKz4bsSG^wOTs18-J%9lXmG(}@s&yE`-ziBldcVHoe+lfGcX~-gT<aM?gVHHUP3VoN1mr@?wWh5jip-#tB<X?@!KbT4bCxOcfeOjFk!IsD45$_MASgSsXRgoC(bbN;3d1C0UEW6d$9?6t|6H&hvtE$Mb9AIz+ZzCc$mg0~kq7n6Y1j@gX6oaWa(h`Ml8M(%~H10}p3g29s2oxftrB56%BRi>Jv=<^WJW+xu5dc94eaSoRegvu>c25=O?vrpZi<eZaz6|~%Ad%XNN?@F+Ku+5E-fr*h6Of14jraEdZvCv|_Y%Fa_nbHbM>~aG?O^2Hjkhpi`OLUixqrGvOSvM=HKC~?qDsWE>wIXf&IiO9S=I@SR}?*Z_xf4;(@WY^JR|BL7@Gzkkx2AOm<>}2Yo~~~V;=5yhSP_gt0`0K*-S|~o^U}Gm?v9r1ZrA>xCqlIB7x~9v8=JutY%o|S4fQE8+t-$j%jgAYfGA|({|Z|CJ#zcKQYOxu-pv?2qx9Hu|%U_I*WtR>M3EF#<;|fXy%`>mybp{)#^l#Gge^KC!D#26Cwzba=lcly78Z-2hKa9iuYI3W0h;tL<e9^VF3r|s}KhWST-Tmqe}A7D5-WQlQyYpw1Yy!BR(=32fg+%B6dIstP)s!zB-75$uTX*35G;GWD^VKnAa=e<ro3PiHapsxpKVXCc*R=K(j<lG%`l^1q7Ntjr&-bC}bX9`B<W|!N67GOxbi#P&zalAgN<s{y?!XXzg7lD5u>vttGJ0SGse=5-*3zRGBv-?oCP|U~aN@NHCjCJRMksF?{~~d~f?Dy=3KgR+#ZENh%dQr1`-<aCUh}!)OAU;vGl^6d_=z%Z{vonF8NWWE*CS4@WuC%&(3J5hH$^HIdJsN%n={TTyX#XTe0GiLA9RS%n0Hh9OLzd7n<@F_Wy?s5*O7X~BAl6MdE(Su`S&Yx9)Z_{E5d!BwF{EG>yM{A$AH0*;S#4z4C)@A<GhU0PCinC8^0i^xxZZIR&8(voQMiWRs9vbV{b-rR3&Du@S~PWWy+=`VONNXj16OK&cKQZEN$$u_yCdKk|*C9E)t4nZS+AYxYgc!<yg`;Jm&%$-Qxu|C6+8nsDMwBM7P%(~N?B%riU(@V&kJ>L9@E18b&G5HocDqU-{Z@LTvTJ2f%NKsMAV@a0VR2ZUu{MA|g#(zNRw2Wwg6X^UTo+c8e!fQ52PJc?&D?M63k;-d=6=XVNXJ{fat)(HtbWWhKvQ&Tt#1r?`bK>!P&yhnZ@p?m*WuVp-JGD93wn;loIAQJ8;iMzpS2}P%N4zz$)f*oZyHNRl3CEzYUcie!0=TJWB4V@>#{ZDJsxr&3RP1ZInj|(L5mv7R(Jg1K_Vc=(hy7PWf?_&Tv?j9p+wlu@PK0Q0iIsYSv5`;a{;P=H2@8!ZK`iZS1)em?gY7r;6|^CK*BAn1NG#Bi`kyR`;@>*h{%}C=*KGJAY^lD%ESBbrvgsgWGoX)2Nzk9_=89iAZaCR6OTt@fz1sfpd~g3HeZNdkmcM~4*<jG+&_^iOV$C(%qKJT&1cEq+(XgvrUWk!Orqa(dShq#%-BhzL-$2Jv1pr)D%7`eTTF{>avJpLI0Z1e*WCemys!PvaecXSo)O#RmV{PdLDg5*^^W=devu(}6S#arSy9Lj;$7YM6ns*-P(timo!vBiHa34N&*4OA_!^y*9J=A>ZxU&f^xSQTLa|8%s8bK(Tpfms?q|4a^d?DC$&W3S3j<xvAq7U!#^l5u{?{B*z%C-OY;oYYe$*1(Q_2wOum%m;sFKh5u)d1`|GB{B$+^1WjU#&R4x1<{QRI(m3plCHKC@ui9J8Lo^+>_|S?JSwD()yw~OZ)i1SyR{C&`QZ;>Plmse);Q5W|#zrXASf9z;QFv$QK)Tof@Z1XHhDLMD`tM(cCCBnr*X2V>4vKV_6+E5o+ksb3#`zNLb*nlE9OOuT5dxrd44yU-%ZW*o6J3FG3)KwRD#NiC`sNu|q^trV4A+j9E%}99#^kgH0Xch!k3a!+c-IZ^Ys1K!USL_IdgKW7T2@qUdF{FXwG_2`XIs`HS7X{TG-ScX{<Wbg2DLTmJfT#Zi~L+pPnCtCrq)@qTa3zNXpT@|Tg%@z!~=<v;~w8{gYarJt{|xXgCXrm{PVP<YJYh}zKjRxu(N!;`Vvyf`66^l8?Ct4j>OwAUMHpYRfOYGI#cQ1#`?Qhr^i7v9ug+3)XfKYO;dmR^7SwEf|I1_{8yLTH~_J9cmA@(miduV33YXmIa$8L25Yv6?~7>5V5T!s{C20g{x)NbJg^-q<*fVI_ul={Ou=AHk|;6NlioerM!9qrR5VM*$fH<bNx6?}T1|5qj$ezZ(j_^(=zl?mgG&r4@FynoO@~$y-P@GC)5iXY2}jyScV6i<OEE2L!(G+N!KmF?^%(yG}^!{hPgmk{MbmN8iH5apU|fEt-$qbli&g7XV8q%gh6Bk<4%W$JOe3Lj)dW>%z_q&ZZg8TZcABk`YKok=s{LZv({iaM^^a+9Q-L5e7O4KeQn#PB}kTCQ}Gp#I;P=C26USLQqR_QxSTW6iD5)kjf0cRRfGmH#smnC2XPLXd*m#|KaTc^$@!SfpPValqeU4Aq@0WWpOvwB=y)>+r?T3Z6#4%+ueTiW@r1^>k?|pEnvFLTtEN&{_CyBW$Q0bNZTLY{C(e^+yZGmd)GSH*=jtx1vGh=US7|v+-L9J5XA2w!u3NK)k4|Wq4vb%K^&YRK%uJ)s2rXyhI{tQpLIggW#R-O;v{y7B}+-$L|-xq1WNqtK_7CnM99daMCMaZR=~Pn#G)HDs50ZLxKFf6ErJ53<Cz$GB#TnnOT4P{uHP3UyXqm!y3h$Y+(idESC`&Dhkmphj5rg$#dIh1CFH^cx8m%*-G0s)lwn|kxRNsxG2VPUXrc;<iX~zaCO~3e>x{<T)76xi^_OfdFpQBss*`atH1){vaP6!?a4v$;j3n6FWo@HctzABTK|@nEHP<6qD(p)97EDM;U!QA3E^Ttqf3?<1-Bb(B#5>a1X>9ElDQ`^g2k{JVKL5XuW%YE1Q)RGK6ROZc-F@tA!z)J6A=?5s)z+&wjN=X61-(4gbPf;`qgKrSeZ1wfg%lPkuewAxbo<}K7-1O^3EcJwUJA*6v?NFnX4Qe&QMq$cta2y{%;~S{wHi#0qCrkM?!hId2P34{<d54spE4HtRJ+{y)NXxvMtTlu+MSR{P=eH#xJgZM4C24HKkRMqvu^X{cS{A~n3UD0%HD{;bhdsXN}NW>29+h<hYUb!`t7^t!o(g`oQ;a(*Xh4?kEV-7Yd1z_qU%4dIQ2&!bBh<+Hn=(&WB7fsKCX`Yt5Za%F|opU1Dn$F|3TlD1*fWe0vXcjJeX*EySB=R20=Ue_L~EsrNsd^M^)#=*<^Z^pPTBEevMIr?(L++$fcGV^;OX4d>l(?99Y1H^P>LNXH7x^;;{VRzTI|c2~NX_BvA-Uh(eSU9KiV&J0EC$RjUow+uh@eb5iqwxmtz(x<3EER;lyvI_)fQfD)j=$_BlAN^hd%4`Fr<tKp>PxGTPKu_(2geZ<Z)tey_Ls?I7-Sfl^^io?0yBMwuTYDcrYv|e!@%j`Y!0ChbF77>S@;-j+4A+VZZ#j0qKY^~VX&(~a9ZS4;)UbbKT7j^uQ#^uz!3PvV>>AJ42>)z4Q(h?LvrsFN=r+-AGh<@|xjK@LmESNB`XSRBJ8o?&PJOf<ff$%<|BzAxZi$gsJ1TVsSIGQC4M~N(nK@egkJglAf0c~0dX`n4IYq}v$OcCy>wVt@EDh!5pw~BM7{J|CHP2D+$?}!1h(Z%JTmMlcwosW&~yeXZzBHA8ySJ^+>dQI{H6kyCn7w*1L)uRF=C~o2(b=>Zx8nCdkqWi+KOT`h7o@#?X`_xhcV4DQrAsNWk-ai?P${mj%!i+gP3u76^ENY8Ygf39;v(MB)7tdB!w3jD#lzXSjkS^+*gkjqE{yOX<{7@W0lb987jqy~KvSv&%0GTA01<Z%6L#RJ>`b%=Fs%s%CSkyCWyKEQOrCQInd~3ehJ_=}knoKyWn#6A9ds2s2pTn<u*8aH`aJQP$dmZwfR5Z;n+v1X-pB(X|l86Amzf>FTRh<Lc8i*qX1X%BVz!2wk3v#lBQyEt|*S>6kn@m9*MMhE#0yipLtkhqT7!$dwuW-pxPz%wwgdO0{Lz+Qq@2@(u_(^xx3)<i-)#Io3eUyG{!{<3oo)+GZW$(&;l1zzvPUWT`h1cKn7dszcI<@T0kF6K&1LCMY0Q5-&>^oWo%D#v@PMz+d1Y1A*?8BRPBux1Z>{<#bn{0{JJ`J{{CsOXHr)0fU;^LyOL<gnPF@a=Bm?)+$K6EBGMVMz{kjxNsvRbP?x`#ML77bFR$IJF~*Jq(!^%d%^g0$(d8Yt_J`zeV_7=h}&>NfTo!s1Vfi8Dbxt0dbFEnkw|TP@WPU2f*$PB<>t4zu}3G0w&{kt?e^X?T!Gr4pBN<z?qh6o%ZG`5w=3$=rQ<`IeU5&)J)Qd;j4Df1vI$Zra-)p1sNw3O~uU$B1kcn-eJS_SsLbUVG})ocs43t=8kW)tGvsh%`4==xy1SaVIlQyQ89B<+>_2t>5KH;%6=Ez1+&89U+GoD3OJiUOQ{$@DBjNTA8xRxEeO0!TpOt(Qm(*Z@g>qr7v5r51cJHn&|p3GQGRz_x?L9-uyB-XdjR!pB)mLS+ei(E%=s*#ur%NEMw7A8W+J`U=?Z~ScV|lTd<3~BPFc9?v>9dkdZ?p&{Hdad;e38pM7#R%HI^VKKr)ZTeOCD_FldfbY<4iF83u%u?2!%1ao)$pmnf)u=mXIOK8zqtCf*=8FkC6CFA_m%5j30n#q(c%GO6ZT^wWt6vZG}Fe7FU)&+V~K3z^^h1MiGvy!ef|60U`T7Of`_ri-!;+tD618K>DYlX5bMEa>bQXzc=o`mOKZfq}G1^9*0!4{!KtW9Ul(Q;Ml7}8Yw8sxjawo00U*ySrasu;u8f*F5T9A0bh&Bylhz5jNp?;X!+xBFb1F6_H*0MWeT<ro{=8*|$<ZmG>gW`N$lG=Z-$bI<yHr18@ZIGJ7sTpK|g=6HWq7cNyIIO%6J4pk^u)QcoTC1oAKReSIG-`puiksS9&Z3wh(?Xs`_nbC0Q;nSziBj_K1NIGme1XsVgV0nv_m_>u!TeQU4bXwT^unZlGLzxX<_F~#b&;Id~jy)yubuvzD3B@Q63A91>2`g~`&QQeD36rXbQn-M+Tl*h~z|aY}zlNw~Pg}i2%qE#YRh8bT!lZS)s<rrXY>e4AF!GOx6WAvKlJ-u!&KKOut%(;O-V~Mv-CMM0KL7aseO{W7drs}ruVw$4jYj2A-qc@ZLC}p{i|KoWA=1(&W0Z1$1oZ6V!JBQ8qn;t$>J{x4t+StNq_b<Ym{PX%LKq{<#tkoqT@@VfHKn>NW0GeNF19#aMU0$-BrwQtMh9U#GFufD5Ja((rvHjV4zaf>Lv)}t8@`Zub7GSH{;&&Tk|f$hm0M5M3ylRDYP<DucX#h^&T{oleRGqe$=Xm(m$PrZd;j7CQtFl~&b_tVv+WNror<$OTBa4~*LU-?LA2Oz_b`mG-3Hqk$K6n(S?Ky=JVbC!8!UqL4!7NoF6bBB<o@~thqzVdj&tFGrchMnjJ`&h>`~}0C)28&&X5~NwL_mA-m4u^$4S$>S8x8~AJMY2xrzRQ?=2rMyAM#5f~H^+htoD`YSSmQZ43ucL)mZl;Hi86!t=}^*NTI5Hl1aefMu`wkLdT?54zu|yF4}?EC0Il7#sEL$Jk$XFYeg=9LuyxWf5%@>EaKVdgnnv8}CKa`3@+yVQDt-kBGywkl%$R>%4ld0y-+rp8na)zqAiSsfcqGXCHx1bmxK7s2ATnfLmZ6rhiup1i!#rwW)=lUl5bAN!wPF8TCp=;&DH<Jm<HJHU2WY@e{Kmm>z$t%q}E9QG=SJ6OZD!TsI5(HxXuRP@LkRw7oP)r7)roJ8JOUd9%-KI;1FbHza<ZA(m`ADd(l{l~TcS6YO{g+eW?R3<Rz|QXC6Q`)+QBQnAC@IhS&0ZHW{7Y4~+Z`Ji^&t-tR-bG_UWq?g6>hjmsn>nvJ!_YX3Q$PzYM?f-hWx4#%vy$I?}9#(cSTHR~2?4$7DTKj%(l1M9Kr|3hX%%-+zkSO`uQHf%yP|Y^;G{wVlIh+X}k^rh-%V}!R?Kbj_vYNG}pDSWOkeE`v*x;^OLU@Y10f8G?Ac%eU*NKAERl;5!w$fY{2=K#;=j~5vJEY%j5FKt12G?l$iGj2&Y3ZWArgWxNQeutjfz%~IoLw|v1M(=ex|lF`%r?(T>Xy-07YtLSrZ_H2mJtq3MxCX9%1ji;x6|HMp<sK6QuT=y(LJoXS8}b|^w(K4VCP_xhw@RNFff7fQD87AhEmpW`^`Iem422c)zwA;udeSrd17AifK*%%UKu>sq3m$VXbA>)cHgk;ieWTr!uMu~1wdqtos460b~qPE65Jy8o2rB#+fdHYHJd8m1zA0?n@dU$PYHj*<}R6E&O+t$YD~5fXVkG1!8Nv%i6lZ@FV<jK)QvXUL?ovZwJLn`CAJ+NzPv-jen}Inq2utW=`4<v>E##4?Kw~AWBS!2f{t2X3^m@c7EHjlJptP?0o!H*9{F{ifb7Q-OtA|C$YX7-*>B;J^rvObAUXVT*;B+uQ)y9~<C<8p0{y;K(W4t{6-Py-@~sD0cg4_BSx_rXWLZG3TM}{Hw5_t#?xx~Pmky^{62n@wl>YXf)+hlXNNB)7pieQD37hI&r9q{iio|g6>D@Y%{<Z45@2_vH*ZhsO`s2pNM(A(!n~)q(`j3Ql<r`uNPNsTBjjug$=G)Eejl|qg;>m#S6hGU^f3{QjY$x|wqu!?bquErynUk$x{GZE8vvM9xil#yWabZH#oXZQ-3*sc^6U-1MXtk)tWz=xNIQz{7Pj>oEjr3Jcr&6+v3B+%#JFj<mI!`5Q*1M*e*}`UKii_luyNa#YO!K;?qJ>RG(o~u()(_I*h?5PDp6h@o3xOw1ZMjz<H=RSh^f0)iLq^ikp%KuI4u7Vx9ig?{oL>^P<!|jYQ_sLe_N5B{kfn4<>roU}Bo*^mcC^1-w;XL89iGyE{4biN4+t_RSvIdZVXn|DX8;pxzTDQQY8Ys}aTEeew}a-&DAdLwRHU!BQ&HAPw+X{1`GZzBcbn%*FC*vIFdhq<g3&y<N}QVGjwiwSEYx)ul|M4J<~Wb&X3F&+7>87D8H@3!sYdO*tvI7`#fgYCn+6lzY238NOem)f0e6Yl`}n&9)1)yMHIO?u&Em0{ehv0pMk+k#GUh|DWhg)Vp6V$~7)Fd#Z@kL9k(k-iZyZTIk(#Mh6~@ZcP9j&5aX9gd{w4wOy(U+G=UA&*!l`Z(uAl4NhxNv4&u`}1D;XEjEu+$`Gp^)|{}FCF1E{_>CuisWmkcowOT_{_YKq?S7o)DIo}lW)A?ZX%hx?z74jc8OLzYjE6tAT*vmjH{i5Lq;)VHDeN9tb#Nez=V9wohi#i%m5Dt$?UxJ8Dc)e9CPHb<{$^!n7&gIr&uLC%wq2&+PW<G%iZJBnG*GUFfx4%duBRCHE#T(~T_Z8<cS#za;Y-4MMGd847GbOL|4cGSFnb+hkMGPVDqQ{$a&*5M|qLIR_fV~|SB$ZD3%AZd*xhxN51ivuED$0*k*?@*!LzWUhQ3K3OQaW<+||B)&Hf%vCtwcemherkp^fY6zaDs{io5K<PzRe3_Lm!(tzjKKiuPI!qTt;6o8W@Vgj&nfAbnvGhY?%3aEKi#?R(-UIEWuL0#c$8J{vAl#+kyIHEP}n6ji_0Ap4wxtUJ>rlXx`8ZPolnQp9CGxy(jozcFLs(GU+iSQFiR*q)qY)}kY;%;8*HP5UD_lb(bW39gX<?E7JCcnNhUi>znRItE4>QHBs{M;ef~eql7uW+KX=S7(eFl6@R_C^r@$Uj9ZX==xsj4IQp`sA^%X>P5%akERA=O!1rweyn9^tiqh;Hk(B`AtMYHQ`M_dvIKbk4!Y1U_MoxG5prsABm%&%wqSDytzxiX(bC%d%uowUpiV)@MsLqPBCLK_%CTIhpI&C9mIDYYCvmzD(aHdTTSPm$bXT~Icd%*HZl+Kn!H9ere_RrsV4Kb@hJBC!dM8K=AIBMBcFrJ@1<=3HzpRCj58DGKs*YouUy71m^Pk>{JXrmHCsmOUIW!$c&j5rr*~)TD}FLNX{<s<}Z+E_K3TPj*<I95#`IveZ50<&vsMI+;*%3?FQ)aB(E;6+&ex08&FT0EASD94EmrR$<JjtGVi<#37gPM6|Qsv7u9p#^cxxNU`h^BO)a@En>|A4eT!B^K@LS2}K?pfW?ktPHsnj?x@Ex*|F3bmn8d&Ithr+U|}fSXfo?atPzVQ#2^31Co=G0xj;XjIQ4(lN%4(eI{!oR8FkB?0->V2S7<=QgjZ*i8!ZZFvk|r29<W6r5Q9Qj*%h>1JAUcenkSx3NUY`Fn6-3Hcc=k;P^as5{HrSla&LlyH!nSTv+ua)bp6tyN${N2)pNcL?#j>TAu=EN4dRjX%F2qoqKp{lH+n)0Ywyt5>aMw@!KIJXfHI<7-HCY&$NXK<fs=h14jec_Zu4Hp^XxH{_=dE4O)tmFTLRsHBmlQlg~X2kOC1Vz^LU7x$3vAr(m|{$V4*G~SAJ4@#H+U--_)Hk5hp%&tFLP9S2ZG<>MOtf%J=So%F!Q=%XWJ-p4ZK7_m;T*3gq5;(8qtmppQus{8L7KvwMrQaq*j}ovLe%IjGqezI{=*FB&$UYLx}ik7fU)pZR6pN)2gQ2OL*}5pN;YGk-nWI3=l>#KbD`+G@X1uX4yK|BtSmFZXJddw%6!y>hSddxb97D)$~$?yXnuJ?6`e%DtcYhxHnJP;{%C&hkos+3g<Ij^Ou{-@f&`ZvAc?dDeIFvKt^~oqzcHM_vDDs2?5u4h{d79^dYy&5_tNV<J<*p;1wcTJ8+^P6zw<1>vjb<+RTP2?YAeYqz@>)1b{l&#>E`j@wdt&IMr4aQoy}csT*i`32SRU_mZ=mvx+YwL7Q^80BqNuI0mf{!xV;2bP1G#-Aw1GP<r%lKgrHl%NdkaO%P?zNka|Mw=BT-Ll6v@F}TWTtW8<JGq|TszD@rUDq6*OJjMzoaf!-b}rf#oB{j?v8!jY_`<YgGS0>q)!F(Qb%4+<)4*Vw%OKG$LiaSwQ>MymuY7XPcV4=u9#Yr8@K&7$Uf;j3yrz4-%jzZk1RVJ~srEVpPyQUTF;t#EAnk3%Ay_Dnr-wpVKp7)sEg`xJ<$tIPZAE>I7GCV_wu|JTj8T(`z8weiLr9O;pwfun?e2`*Tr7Sg-2%hzsiYM)+su+MPHy(njC$>$%av$d%Ro0#(}g!KO6F(OtE#W6gUL*XWgrd%Glr$22!|#sM(_oG(X36fvv#3bJq_&zOM@tE$t9h+kaExqCR26JxAP_po%dmMdKQorJPJd<=07A+AyP?>l6}y9wEie;JZ|(FYmXj3>OSf}?)5j;>z&%8{zfN!yzc*8d%V%95lKjuVN~YzB;!Diii)KWc~3G5ICV|ZXpk0Vm1>jh!p`tSN<OelZ8U--2T4ZNn^z*Th74ajPq(4_PF%}e`I)PF=Bi-}yX>nmAd#<JyWH@f)$rd;&|agP=gxiStz7yd2hSoP@Oaptwnr{?h5q$xUiw;n@oNq1wFa283-GneTFu|Ve=~^3h*H~Fo9S11#=lq6Q#SVl@nWj<VntQ3?((7Ui6kLdlfsh_2Jz^s%?1EP{^GnMhEU;Ubl+)t&Ej{GwY2byyaqBI!qI8*or>fyZ1I!(PQ#ZnC<v)Fv6-~{HM3w|(DO<4Etpd735Peh?hE2oD)Pozg=+H9an-fw>3|&jyCDdITYf+=2xXv%vti`EsNmTO-Ar{M^!15`?>VjNynBdyh#oyU)FT0-tTj_Jkz}PPuPThNspXba##9OkimN;RYW<!BtPH1{D)pGFHR<v}^+)S0t9(%oe|Td!4AgzWHJsk+`i?5!%1-13M+R5%Hg(}(<d|`FC!QEN)v+Ww#>$Jyx>Mi0hcnexep?iu*+0;D-#=)|CvaIh;l&}neAKj{$c(O`I1Mm}nfdD^4hCldZ5Tw(BY78jV65++GG6J=6SUhTUGr4YSr+7QO_9|Kz55{Thl5U>Vs_|_CusQ>=rFy3TM)+$ur(6MCspmOL7MAu4XhlQy=^?_L`V78Q|<izhmQlasK#J@X@7fhxQ2m#;Stj=zDmmmim6;*<*t9tUC-6^?@bFfZQX0if}rj|dT?B8Yxi9Df&^}qUtpPnFP;!6c(6P|H{aNFR(&r6=c){jZ^QBPaQrejzm~zN+i>c6IQ23(^D;OM8%`q+r%{Ac)bbdIQ1Mz-SkQ(j=CpQ%oZu9%zZ)VFN>YJe8G=Hjl|@iPp_3;4Ad<kJ6+{>s(!e_*?_ePIl$=4$o|<aazUtUf5donYeV7?~XpQ@K+lHOSs5LXa<)mg>gR}(r=M&<&>gUQ9^VKhTCNWHNp|NWem<ZsxLxzMdp&8GN1l`MgVWiPXh!8;F<qR?O?I5};NhApMoMP3bSdi_D#e#m~GAK`2A8!lTQjQM8BaErto{T45^^QhV0YLySTOhTNnL9S8r_2<t4!y-($7klkXXd<Rfz8Z0^yqNTv&;RrCsMk7bchE>5@dvisQ<X#)P%|N5p_qPx$*R-K490U*~PX=TQB}T*#eHf2$y2ya{|X~Qi6K4Q*lrc0m$JV6Z(6ebYMkC?&@wK<8n)$t*XumO~X?m!`jN+hvP!ur{5Ne$8~e-D{!1eB>|$L6U00!X#+<T;3j*UT<WCCOshC)?S8~UdAi&_+l`eQ!L3o7_(mnIc}-JDQKD+(xDvSMQc*;u&Y44{GXY@8T2pb5%6nv|(z$63CQn3`Wv{RhqyCAa%-sD1_g$tZi3(Lh0)Fct;cz(OHy%10o`grz_@6+ikvPhx@sO&4l42rh7m^~Ye6*JPuh=!fSLuJh=>BlHnd%S!-n@PUDpeFeHSS?nQJ<v@P<iLTyr}Vtlkl(P<g3?=+5WL~{2SIze~6V+aW3jNyqNHd8QH>GX<sk^tTgITBERa*%qEpZ@&WUvg;`nXRRcD)D7oRyMBcR4Bu^mqW>ul{gb{ndbC~dL#%%_d=dwP}5kuKO!2&(E5ZS7pyf`GVE6zEVDeo<~*|KhesnAc_sU^6V9zU}lLsh_o3e&@?dC?XIi;J(ce~IazbSx?BSW?n4Sw!5iB-1frCTYhMpZGnqL@?phw$*H*NzEU18=8-1H++kB)D}kXM<=^_nwoss;ze+vzQ$J)!H%tucS1X;CRLS8@CsKiRHX^eXW=aL3^O&igavMz<4GijT-6m*XB&{{jbaW`d#7`yLKu4CWlC_!z)QJVD!+%2SD8~hL<}jepYoDjU8Ua^eVI@CHJRsrjR}!QPrtMBPIf6_E5j?c#FM_k2I4ckq5*Ppwky<0Qit=HlNuWw2u!^yKrm7e47^x}9n^H6maN1xTd2hppFze^K6~J3MN2pUg<kkG#3G~PEE3H;HXimZF#Oq;^_^ED0H>r?CNJ5R3s@fh_lcRMS|K<_PQuIxm<uMLnese1>d!^Ohu_^BTF>BvA6n1OH5*Hvz8V2vO1?5EC|}y&S&A-!UnG>@zn7&gjTPn7RP3<fiq7<?Pb>B5iVwj>c#g+y$A`6Ql{8ikiyEk-ksfmCLBNRMNs15a)oMe(v$BE8$@V+gG*v6pQ?L_i%i+f2a3rThz=uq2O7bcff|HC-m@5$1{+3qEFoM~9XBVgObi1>&8GSpBABz227nGCY<HInimcZg@3~)+asZW*0!*LHdHjd9&#mO?<2p=IMx3WOF0PtI+uWrBJ+thy8cFcC=5BsZp0xE|}#e7U}Vh-2Uhw+*6Xg3&}CCb6Uxee2R)*z73rcSGrKpbp1?uGCIbZa_@!n)_@?=g}x7S5bsnLbLxE0WY#OoMan#!!wMDFu~GbNa4r`^;uR4iYPRL*KQ5%QUjtd9u1*?`K>N?X;(L0pX^|cc}*z`<2wuJxIJ;IQ0Zo6I7;apS#&))~wItCieiXxAu0?!ps{p2Xx8rh!#afsl}%?6%sQZ=}8+T-AXpbI_BOKC2+HI9LrvPjL;oOWXBn0e0s{l&ndR%V~eLvVb4Qn3S2u&e^@?ktltuPj1P~Gm=gdt{i4pNj9ZOontm#vU`wA(3U{w^A41nQ6(9zLJiprKt%H=QzPmb@vV9!>RrOgWF?UjxB!Cs#Tu_0&J@s6f1yue!^&~ZI1(@d`i7aipnjS7Z&ueB4ToOAYq>)%5;Z{&|CNjLPN=JBU4SZ}OTsmPsRxI*%)V1?<)D_jPA+B2oP-ySwd4`h++<8qTGQIZrFU<kx>>NfM(lGb0e+T(ZM_%(gQ~E-R_U}y5iY#k6%JtdjJSQAq_=QJTc%)^F<eY#bumOzp0=>Kd^StnHSe15C_@v6U7_K4fS0HQNp)YD)^P6CBp?!?2WDVSy$)2=vnkBtJyWZXD)=-<=9DF?w6s)fp&HZv5muc_kJn?H@^DDmE_^Pjvt!6|(%>5v>q+V-pDUQ{)h3Exf=tz#B+ooB#uR{|K)A?3!AE|DcE-9Dq_Yp;{jE_xq*yO4z{-KiTxEd1L+Lk!tEl!Doy^7=qRqsEW`Ia7M$;p_ncT~C46=w|>EK(%Q?Q|;LFmhCUkhCpw-AA2Y%cpNh<U0qr=S!rh98;;h#|LY;C}Px^jZSs)90__{ZBfN1py2L?N-N=9T~&IVi=|+gs@x-D6V&NGkNo=TYiVDdHmFaf4p2==I(8g#xNlm4s9MOK-rqZDKU0-&8}-NQ8x^_NdS;44@b%8$5BMGHgBAT|QwZQT!HgUtGb7$&hlU{}MaxjP2mzZYOV2P{#t4DcYiA9s5W3UhLt;_!_A|n*vEg6VT`fe`|GB=#r}D8-e;kZ(!X2X$Y7kekmZRrGuGhTjxD1r^Px@@44&_VRC7jZ$Tn4#^b8x1l!3SvoET0o9r=M34T)eRcNRMgUogtUmQfHbnjH-&6!Hnk1Tyu$cZZ<7(t%`OQ%3Q1ZU@1n>x0!COtirWbRqNSv7pUW!S|my1`!>R4T)s{GzD7K%cDYtttIeF>Xp`tjL`coAK{4l}T4prV+U28XwHp4LiPi|RwvL21r?JLPntLcEw%k)f8ZYe?XHrGBBErujsV%M+$u0_n4JV;pYdmn?yx3MR$O?P&kzdk>)bWS!*K~pIgfm<juh#Hi>+x5dQ$%}x9TMDADI=XlgEj%(7zx`6;x_7Ex;o|O0kT50Ixs_h+meDhx)4YepQ@n8&CTvKsCL%ZOhuhcu%lj?^rfew9R>a^10K1jX!RoWvz@)Zhw@3GkL?O~b8#2%UYCe_$T5;RTpIR7t7ka9x;E(d3w6k;5=V8a6F!R<J#;>86DPd4wTa(mWYhSHo2v@g!Vo8w%uG1{=Pwf!3)aAJ@ApjT@4|`F{Vt%y4kTsM>ieBrgpy*DT@*ScH&i&3kl{ropWQ9TKa~+dYVV6X(DvW&IKN1tj!iD>kc(mxD?3};b5GbwLQhFJb{+y4N#h*|&CaUhs#{oo^vC*A<XDmnP9+e4rCRn5oE*u^1TgM27iP34F-#yXcY9Ku@64#*^mluPB@VC;4A^wvRO!aH^zoZ5eH`7qJm4AZPTSMVMGNIN>pAUFo>UfwOhal(n;gv`M(ChW-A8($ay_Yko145rEobU#Wdg15)$yxww4AN6m5FjU|KtY4IaVqxo2H^TE)}Mo<rKP>+jn0c?O-(e?n{J)^IRaaXEpkSI?0ZK3cv~+eEls$;)573NfR3+w;Ioh{DFZ&enEJ>OOAy~vsiQ7`<LD!Dgf<2dv#P`5(DM2<2bsCGu4;(okoryAYx!k{CFo&Qq)z8uK3M@Qv^ve152ZhCG4PJ<;vyRlVXu^DZk(G*p7|(O=C*Z>T&|rDHbDA8mtG0gZ|{weqCG`^3jp<azV*%W_zS-r7$a65aKseF@9)gMaO}<>uZ8&Fe@L2=}o-_PJ2s@?1~zRj6eDmiexwA+PmcG>mF%tj%cjz0Lv40GlaB)+enO-eY05r!CBdh!AMSC2`GM0@Usb`zuaYfiVoEnFAsT891=NO921$U!MTz={(JhG6g5?mM-@qs0&ZOJOMD{wJb*gR<rasos~jSnj*kK#T)qdY3TYAFvcml~TK?0Pc%fE^pD5B%Wp-C5F^+K#<%|$jen9EiT<S!tvWJAn2iyZ^g?^G6upysZqV^ll&fWMyCdg2N0^k6|Ig;yZD%7ylnF_u!#Ns5?xXZuqJL?Ta4#LweZlW464p8mwd5&Hci<rzt%4k?3S;SX1{wUspNx^zF@GgoAIkR}ZtVgvxbr8)E(J-nW@QN}lWlN>6{4NQQ8Gotvjfy>~&6(+Z(mHJEv`8wX_>G1gB-Dqx;;hsq(<gp%rL=?#tx3MQ<Y7x1bv%nCb3{yLoR!W&Yz~K?7$B;OcYrKtIca%1?220eL@g&ccg8)KgyaZi4uu!`AI52oN<RCXbalyLM{}#(L54-IP#+k1qcAW=3|%|iK_!e4AKJ?$egfw#jA=#9$FmV+ndDT`Yv2W?*RzK}XO4ibiqVA;bt4lJ<8F%og(($EKvF)43%?*K;p?2Fl<>><xs=599I3#75C@_15WACpTU(R6JKDvz6Ddn`D?kktiI*9VcIdl5BKyPWp4!wgGqw)dGn7q)o+`_gvRS|e!UkPVx%5uC%;9%bu%eGxVm_%;7picf(61`LaClJ@i@y+>UJ+cN;ZP&2B3Oh=YcwKmo$Fw7Y<G}LFm9Skf4T0^RKue}f0}ClkZ@2$%fNJ`?1Z$Tj4!X?sYXc%Wv|8Q(%5ie2OURJ(T@w4>0}Vrb=0RY)_Usr4?90b#PF<|)DNR>Smp1OzGP`<S{uu2TBdHS6rq^N`rI^!@u-5DNyU_Y{3($-kJfU=LF7_)h|6eX>aj>5*pB8~D&{K=cTCWi6>CL{K-KtsrVg#3386o-IC7J4Q3DKKE0O$)a&J&GdQMx_w5AKqt97Z-APWa|v*zjuIJW7_*=L(sdb1;ittue8qNTChDwa&Ft3QIrz@!QznNyr<E9fPPg+y4rtE5+)^v>u5Q>T+}1ea;t4`V941i5O`Mx+lDcUr(~(w)XE)Z_1pG7|Vw@t9TY8EnBM7vLbV`$uv#7W(!jl?tSrPq4DINt|!3tk_{jm&fL5E;x%>7~<Y|bR3QJi~QcS96ac0C6$@23ART_TkXhd^}-Ap_Ikm_G@g;*WH8yZdZjmZ^{}LIQ|pGu%rn_!2ZtQtIMkGkGfLC4sQj`lg0ydcff`m4RCbb@iO~P2;5l~47#Es3IbmO%Z|-YyK&UDICW1Jj{%WeA|GWHK?q}o}D;9+ArSq<KVVQX*$We)K3H*aXGErq0I%sa85r7%P3V|fiC^d$#Ss>y29%59bhEFm&a_j^H6aNNIfP85%iYK(bEreR_V#ZC&#142d?cAyhDcbFx=n6q@3N3wgWd+yGyhc|BCmW*od$Yj1bKX(N{RCXZe(~8s?;DoVQ8HqCA@(57ExM<_$YiboZ66oU6MuWV$y!<ARc|IqTMd|q+!UF{CrrDw9))Me3934%#J<E8iSMPNos`Iu<XWAPsj509rmF76Fd246rsUX~b6GRSnzUZlEI!HcN<^$|;jFntqD_l>AUzggEL9;kM8wy-gDxeerQl6dG|F*Nyw$4i>S>*knFH6EgLjszZboWX365#6Ny`sL^Df6OSNj$G_T_hOHwy;?PL5E*oTt+{wdse`B{+B1uWytr=NX!ysd>t5m<V9gaShdim^2+75&Zw*l~Cl#ap#<CV<5yDU;t&tN!%_(yw#MCt1%UVXLNoXhtRNr`<%UVD1$#3p!S2pz#YHjxDPux0b(P{aT!4)v)XFS>;g+m5aia6twPGGDGHz{FecT;M&t1*N=%ip2o4TY2rpT_N163|!3h5CH{MP-?EHd|EVJT~tVqJjskNYJ2Mk~j?>R2qv_h?14==54reqhEe!a9)_qWt+Hy7lalx<ta!BL!eP12D?(veNl3fYm(C$;sG?+#%@WBh|q`C_w(IkCXyu~)JdIGHqB9PF)V`XWl{zGPUA9@1A)bo6Z$ZDs|-J8}N9#eB{ljj+8F(^+A9u2rp{kB?I}C|nmT$6vAu2$>ucRfohX<1wj9ExQ$m`XM#ZaZW-MMB`*cTD!Zx=2=@tX_&f`KDUhi);4ugQ8quZwq4e;DO5B_FJnx+=2JiM77_Rj8){BBo&<5GJdo>JpFTAE-u;wxQeUsxRx1GGm8UYKJ{g$3N;hb$@Bt1*fvRp2bOPDJx;V)r;Zue=3L$bf#N7LufDNJ2fE(x<?*I77ip-EB6E@62>3H45f9jS}tkmgK<i9<*#zcn-T|A<dY}Ae<;}TQh0(WAcX2Mg;K%U>1#~WT@M%Jk9tbvpI0@DZnT&akN89Ns8`HI%(4{k6nmi0U}HQHVFa><Fp41;c0l?q^fnEBuD6+Eslex4a~L3W8*b^%ej#y8~|^CbIKT<xXuZc2*0-n(L>ZGL;Kxe8M_9EG$vZvoPB5mfmmC1Pt1?G4>SSqWFXB)`xcRxfFXpg$aT!1veX{;zU>?ll*IgiyutH3YnZZxylCCD4W_X|S>A6<}CiYo@_EZ$G|y>BgR_wMf4J!plD2D7>)YWk|GS=wLcuHFTB~-L%^}hI1h{s;ItA4WhzMzkg&<!1oCWfQRiw9)G8yr)#FrtNC25jyYEL9P4^s{T2V4|GsD>ox=_OMCl0KBYc`(T-g`D+86Vi+)DaYb`aW7<;GS1#;^GsR;Srqz`NK&*PyjL6RAjmzeEPCm`yX=wpGFo)tyk&36-4UM$>)=C7}qV$rM53nb*9`d#3OqygE!fzc8zU^#%znGTR8+R`cq6`er@7iAsJQ9zHjye=;iU_KYNu7n`$;hn0$(I1kvO_-5AEyygNx{v83aI93Ww2u@R6my<>4|5yC~U-|!Y`oGWjeZKGWeP3wd-)-V^BMX4?bg~0qU8)%m>gF+~J!qtl7ZDVb4{B$kWF{(QqCn<th$IW6UIN61atm5Z(_m+Oge5wjx!ADmda_V?H?OgkjyLYwQIl`^=`CN~s;9T=>Q*DY)kup}=)@LmZ}(eO97h0Gu79MGGr7hET3Y0vWB}|3?Ri=rl6$Unx1#!3Q5~zO-e|Ps%{07ulhlcD^(%c?Zm*f1GhkU`@_vagSwqx{IqB$Hi<Eiq53owvmBpYnz!qM|*=58ZN_9eZ*sH2z>lMykaCH$fmhDNqh%DDyvj~FCHT05-7MjYH4i-v<!^G&xNvb8^5GFF$B~h%JDD#pWcSTp{MY2NSl<4S?V4*sawKfvYm`7TLebbkll3`TT-fk7@KnyRGs~0-(A^R=oLUE4}8GyAromy#8W!7~o0Vx~1ki!9TqSzfWNVX5V;gQb&Y%X{|n|D<&&$~)$n6q34VWHuCE@8bim#|*F%7C~rF)+y`1;!nPx541L<iOixCwktUHs(BncNQSnT3sM{?l3EPgF&+tW_A)9cbWvdYZtYC4%-dRe}jn<e&gcPD|zZ$0M%kWXER`*li~WD4adLze)Co&c|%HDQl1jm_nO}X#N@>FOD7gNV7EvBvog)@@Xf|od}C`UD$Q(QYUrkzElUzpgXh#Rue9EqF-^msXROUI(W%#0R@iQwnv#?7T)h22Wcn=6joNMTA+saiE!kLa+$+U8zo1shN#C-=)>1?4$tb<=a*Txd1~T47Ywut;<VnAYA5ZAl2)oD0Nh$B-6A+Y>5iu4neY1+jR5Uwed*z!%u#9=EPb|PI8Pc^puc`g;1W;C;;5relFaWqh%^2YSG{Z=(;wI=IKc-XOhqS{rM{B1n<SppvDSI!Ct9DBr6IohT(2$5WN6VCFViDNyw^QQADScNQ8Try166j?)C9F`VO%?B@vh$*GHazA>=+u8y=#=fDl}LV$5cm<^^uIg2>Gp93w|0uyKGJLs%GX61l*fN?!MTq*0wmv1G5o`o54_u|kj#56e*Gh^;O4l3g6H%Cp#w;5`&aq=W$tqBeMhKs5qVqg{Kfw6+hz0J&gEV1mrh;X*Se;)uRCq>hdcU*JNn<r9sMWxRll!4)j(?80BY@rYxak0_J?cshimrV(KU;9C9WRz5H}sU_ES%-+^EZqhPZCme)w*G_-_9reYZC{)~sV=*3_Xf5H?OzuPuIx*r0!{(XXs5_rJlNTkzX1*7EP>xm~2`o91klznJu~So6;S@a~ial1=US;W7T<G5$u6@!I$F7_XIjjMsj6jBl8PN8iz7{ODSb@uMFe<NsWbafy%k;~(H4F7*Uo)U=7xb^~iK@WYJ3@J=4!Kg9ogOYiT$%#A9F?)NDl3)d2C@|$P}CkZ@S7h}CTe~fmkOUIXcX<gU~O$ezgy1@4L?1-reD~DpjN0-*GH7^a5w5O(6EhvWw#dNId?4?IRIZ_o4S?)BP)74}e48$AQSVvhoR|Y8+a0w{4sPj&+^f=qIy!0))N?ltO|050|_RKIOhqP9$*0p0q`Rw3$lu8hn=}`^3uo3qTqvRi)e5&iqsrslS>k)tKBYB1k*pAfTq`gM9x>l%MyWiSXjy?L-{JNS*G|WY14%tg1e`X!|dz##)h?{bdN5ND(stT^}Xx7nA#E;@#{+RVHm)9Hw$X~3|gOMsd7_q6#-0rK}eb3_!NAEZ=E~4cX8jz<q^;7n$lvFHST!>CFtbcL2!hhR>g@0(t!nD-b_vBKE9GlB*AYwM4Dq7<NT{zv-^4Em7P**|fT}1&(ZScud&8(XgtKF35K*<g~rM*Aqb5$bd2{DdVNyaG~fw&|LSlY+EfMW>+W_r?!9G#^@1r#gd7y`o!^YH^YAFg7gssQ&lqjEktBR!BN@#k9LZGyQ2N2cy}#UY-2>B(Kp^o`ZB!A}S(y_9s@$(;1y^hog}==EY9boF}mEI1jXpa!YRJ5Dkm{`%{PIM)W3Kpzm_o+OTsO(B5c!8)$QRKfL#M*t+0;59L1H+FYP-atKL^AguaK`G<RUDvPG*VKoool@uLKK3xWl*3+rztd=Q<Fu;tjyhxOD(Q@~$FJORJg!Ud3a^xo(M`(VY}9XAd6<T#1W<&zWf&JzPE{gpn1S(W)(9dghSxZ^Yl;@>*>JkBEqUB#(zsQV>DI~{@6%g#f}Nd<x`2lz9dPc|%$^V}yZc$TU(%I>gQwsXv+mUS^fDs^Sofd!PlZlYq~48XqR#?#zK?r9=yttBeInk`;o4D3fcM#$YF+u6t9s_DkxB^;h`f9oO*fMTWu}vGI;loswbQ6WNTC*Mxm`#h7-nA;v1Q@6dbYk6ZJhe+RZ>c;Fk0~c9Q*q(_iEUiW!Jmc_`TwMxwl5Y9?^g6^xtE;b8n-fn%6z}J=+VKww3;}vYQ_Au`c!vKYyd1ztK4IQfPQYJ7iGv^e^B1s+(U8FVjBlwv6a>+gwN=^&j+M(TBf&Cjz1W^5pRa^l|TJyv48jAQ`{@C(w*_!_UL;^Dz884F6B2yT%W?v*@nA_DysrTq#F&UzqIf`FPvM>;5CA#r6MBXwkY+&%>zaVbt?5>VGmlul=BBi=G?x?@Z5%iRP*K3zPFbA7A+R#QzyIUa$QJ(YSS^k%!U9!)WATH2&mu^61ApvDV32?N3-Iis|Q9$roNG_v#EC-_#qtcGiBZoja`^>&99h##$c6S{}yQpS=Fof2==y{cZe<)}Pv>@~iI)FTZ<rd{f6abyhj+8$Z_MpS~uo8;|lZ9_3*?%ENf{C$H<rKi0Lit{<)c8?9@#spePp7hcx)>i7m_x9>IBR?w*bSnK~^u666idLG7l9>#he#`>SkFE)Pg3yWW@ul<|z3nc{Pxy2V|7xx<Yrh#u7Y~yLH|KK72u{^}O@i-6TaURCwJdDSG7$0J1Gu+=2rlXwIa$>Cs2jWC|UaoK?ZxLTOpW!&+U*Stv@jnO#nZ?1#Ih8sAUWvnbOU2^wBXO&9ScE#&TF3e*2eK25SC#YF>5k)&?oQ+3r85ey#<Qt&7!jL0rT^sEEyCftvvcIQZ+6}(&vP&uM*~DtwdfWfZYg*6y)w7m?SgX<rZJm_?YXXjVG3U?Oqi@JNDWNB{tz?AItb0$0BZRiz<TB{+8rD|#MqpHb}2DYo6g~B$DuW}zNXO&oF3)8`i{#HTuv7h1NVM}LovF}KZzck`_5w|8iS_7i6h5Isn@#>Vj&XM=Bb2*;WmBE$3l<$ecD2BJbE&{<^aaB+8dXMWir7lmn!zqJl+O}VjQSLP^C+9U@cLU^!Cz;J4c6oMAL`S6lMFcr2#iri2az3XR4Yy(KFXjX5TR!|3f0^OoM!S6^GqRY3BZT)YAuH>js>lk3j7LAPz^7lczZrMM6{bnv$qA7v)^SSrU3>@&9S=s7@a3>a><_x2)m=u-D~W#FB*GdF>8)r8=<`iC10DD{bhs(rlVzrUm~lEl3jxQ@&d}s8(J3+G^FsadG_og5uO@%Kz%777Q@k;I%o!H>95F!{XIx^vvtW>R7io9}hO=ZU>a<cGOYx$3(&ENE{qmzv|a(j+@Ln>O6QkPk0!0N3&jt6Uh-x8RGbOt5I{tdV=W7`agfJI|ER23YIv}Xm}dxo-`#Kb>RHw3|2OrX&hXHBa%va_d+Kx>kCp%U;m5*x7y|U^Nkw*o5{kLZ(KB8;SP8UzMp_nBqLVk#oEWNCz39hNWNh0vbI*^gVzZHVs<#NTtbmHJc-t+)pmFB-;D2)fJhzm<o}DZWhegiUGSv~{vX(dWS|O7m}Vz%=-?2^OTXsVW|Aws_@zd@QJ-A{-p_wYVlw{AfM5KQ0k4ypci+CO6>^ztUgm|3;u{;eSNy^oehx<QjSb(>(t@<kigQla=X8Be*XL<+7g9Xk`*Wfx@A^DX^`dWH%mPeb+&)ZbD^p38s9C$EP*bDLIs4F84}lhc5rldQp<aN{uplhx11EEue)#Trz8~^}=?lH#Uk9n4LaLWSYFLn@#)4#rVhr}hM&`=TJn}P{AZhg6H*()}5s1Q#Otx%{<z<JNK3Ki~Gq$y4${SpYkxwt5Kz`laggF3GV5KnSmBKt%3X@tX%w&c7v$z?*QaY=Z(y6SJ&ReB)vg~G4eCrTGD?Ns=@^lL+CI+1<%I}H))U^(CqaF8Rh#+E?^bl*Ck+I|q3@uc#n0FRj;jF1RL{+kB7~Lgfctd3jv)K2~EgmAt!!)#0ruu(o@A!o~^}?OTQ8S+mqeohMx_}06*R@QqG9AkFCDRR%m>4J|DGSuw4id>6bQ$)2Q7U8#f*y&xE46B^pGJz`pmA<Gbvs)GQQs^FT|!fn^b=Mj6P!Fqa)oG>-7$&dT-^|cs9`t2Nl0(sZYSTa@BIsvy>Hd@zVZb7Z<dmkkem_4H@w1`H5<hy{cjV@m9SjWuuARvHYMqgQiMv-^5<wq1}!BOeoAG)*0bksH!W~eQx$Gl{&l-)^_^58yzLj9F?cgBEP*ix*;kPL^1j^V3)2^kp8U~4yuEXVf7dGAQ(m7dVR~7B1Frv*954FP&wXk2`1&tpvQ)94TBcf1%`O*IE7S{GjBW+Z&p`8SXnwK!kX1XVoI7%nETjPrP#Noed-p&KJzTYgt7J`Y6t$F<h5iz%Sk#(WvVm0%1r{JM&A^vmM#hoqtAT%WC52;Y>sU8+ydZBjsdJL&1N6!+NpPEbFBw&!er?GY8q=|osYMaNtI&3u>XC0f@|BvSpVh5r_3UR2>rq30wqSi0l#&~|?3Z>Kr|`<7Vwclrg)XO$ie2{e=W$8BnCmAvQi6K~!JXs<OmZ$Q-)7F8i`x_<LMaASs3N8F4W<qoHFk-d+F=CcE>I$u{^grrb@L0NbLRq<0Q&#cwew`R;HwsV(}HhW@J$Q;f)+ApUDS2oblo>y_f6M*1Nc|||NJYu?whXrrt7}xdU*?Zim)YY7E}uvyegG;XYb`L#YagSyrx`so)dOveO0k~SLgaBe?+C~+X#Yug!m07h?kW2MvadAI>^38D}lzD+T|rxx!>V!L>1<FR9a1!i{2EVXWx42s+vDY$hM3ASmRHVwqLgh;;ZK*Ex&Gwz9MHEGVZ}ipn@=xn`wP+LWaplp79~%yTvNnrj3s~t+MQDZj5MqhQbX&Lh28H7SI=Nay;x~k_bxB6WypcycqUEt4R1vRdwLQyidlR_PB2y<~^IKvg#-5FLJEyvb`rLRWX~<z^Q*thcm$Gaup9A$h{MukUsnK4o}W;Qc>C>XUuI|Ev8lXrJV7mBh_c5IB_3cB#1iCIeX%3<^Tk%?8VJBUO(PY<F!y&<M-S#=)<HfsqvX{s0>b8>8K$AM)c|B+r51hK;V+SuB)rot~ifb3~P8G1oLJFSFGZ;FYQLs41WB*EIoVunxaUBdd1O{3&!FT)1%@`P^!21-3O-FZefkFFop^yms~Z=Wo=UJSH;7ip`33CWO|pD)5FvY!X?BQzwU{bbvk6+h&|);)y~~g=i_F2tmy=f16UDIs&t2q1>my%(XcyR?Z;u58ep>=>XIE?H!~=!WJ#u`G!ds%m8Nmy$I2E-2MdJ1n2#|=?XL>K<%>+wuS{Kd2oSN9s42XVo1!G{=IY#F4M|Z3tL9Utu#bB|>?dZ%ZE8QACiLqyRA=hl1FQP7s?4S#UfR$AhloFnrcyK~K2p>V(7JZeHITa#iT9x${6<EO$y$MrSv6sdpfPl<`i5w_^cb#YIGtm0I*XqQI$zwaI2XmDGP#mecbwsLZXX(ouW)f*RGyEE>j3}!8%**aHph#nn6<jt*6Q%?;)zk*Ws<e`{;n(KgCC!UaTJa=F+Hmy1F4_|uFlZw#c~jZvfR<`8FfSEQXI?T2CDkYI2v7vZ4&0+Qx$RGr6N*+FUH-&jiYQZ!DTO*7H?hzBU4|9xqD<Tc{?D2pJaD~Z)E<$gWW9m_mrNIqs2_`9O)arf6Bt+FI>inCdkFTRlY}()PXJ~g_W?$A787Bml&N(N?57Y23kV>>4TEmei;^8<e)kJsa7pE==>_v3y7!j^yrR@C`;kIaBM(#Up!fWOhyc=uv9VwxbS59e%o^V8JOJZ0-;6;921Mi%0!JjZ~b?ibi41keHnQL=Ag!2F$guBg@}+xOV~T*;+e*KT+Sc9`@u07R9|FZ9D~Cgh;@s}I2lfHvW+VAqU*@$S}xkDQ8OIi+(ZD`QC2GINIVx|tluW#aMs}-h*<B=Dsu1kt{#-2l67ZFN||D<Qb@DHWu#6`g;du?Ilxn#G(aDFxpJv}-iYltDo!fdabHCtpi46u5QhKu{=*9h)WY-d)DWkOumH&FYA2H0E{#+kFi?^IQR=mXN6x^!aK_Rg9`>FO&y<p;u*h5&!;CC`$0}Qd%Ca;@joygx8}XY(`PQmETDhY{j;`M<lTW)WlGDpVValTgC)aA~*CN>ldej6Zg#eI<mz`{@s5lh%JR7W4@5{_lq20IxU3zA!Y08llamVKxn_T-be@H?t=X7EVjD71S4q+Y2qCb<PS-z$u^B3vT@JKi(3gAkQGIxH@GpR&DO0Y^^^T|i2NlpW`2sd{sT%wL@p)b9ZkmWY`)wvWsv<}+H)Hybk_r)w1A~~}-qTK4p!m}%HGQ>O^EU&h@u-{l|-`aBnD&kf9H61p4`NVSG7bO93Q070V(FF<ST=<ZAH9gUUnHA16u@6K!hx7u*hOT9gY@rqM&HIHMGp36OUaIrV5LPFZxJ<ljSz&Iy9jN--wD?tgC>CWdR^;v{vMdy6&2EorQy#2RZQnwZMOdV~^AT0V!SWT9VF2<sa}m>+*;Yll$>9+fmLtw>30#r55;-xCaMK~Ioi+=Q>jlX5?*zF~fZX^_kk<;3*S-_vM+L}_z7yp20_64Y1o?3R^5buZ+zb1`Y&0!^-uPC~*Oaop(;B$t0=O;V-;(NYOYOHL_1lv8E$REVbbU*rzAZ7|l9F#r!M7yeTa)cAY4%!LU9{QgQy_w<r|RR%=A+MpNRLCfJa1YSTYYS1T0j>L7CvmSBB?t=RTfiK+2!(<<0vfd<)@IyS@f_sa28Y|Eu<9mmQZOd(c(iL%*KzcDyyz%WS9jyQN&77gf^U2^x_sRX#qRRpNa=bUY+L%HIWfBBPre_Z;)spRbVr+yImDum(WFR{P2jY=;9oL@W{)R$+v5a-^H(O3*ph!%gT)W2J0m`&PKx`@oxh4hc%)~?(l^Q>klePfh+6_&B9rGt!wvop#36f-+<1!qs<~Huex619x9j-%Bq!DQXx6r68?C^>$3!E?AK1t5x&yQqi^RHGb6_*<faA?Yqyx6om<RLz5!o1;OCAcIIpQVQjDf{)Z{z9x#LUxkag@@8m2?fOJ<y4CT$YbXWg(JVAihbYl4vZLCM($T4`k4|La!PbfDb5Hsgh(covDn^d!>G)J~M144{>pnpSSUhj}VPsA8mol2lsP{ucGfqw%5uhITakJ~0d_$R~)V;A-Uh2p{<l-^}E|0PeL47-nuQ(cJJZuWqa3$dx^gRva_hAaRoMIKp4N3zzdWfJPNw64ylvlbk=zfRquBjVg!{&L4P=ldi88m19?<7nVyivAf+m@Z0b9o!@>_PT*Er2{?^Km!qP&#<epK<4^|HL*AqKHKYzTsbPTURrFp;;4n&%wTguv&!fats<zjGHel>-AG8j(5B8qP8LK#u;i)*)ISWyj+Qw1vs#}l<2yDKLzI7l)q4v6Urh;W;vqB+$&sCalsT0auv;=WJD{ZvBtEi1r*<Jg7CAH~kxkZX1Oc*^ffjECTKmul3+na`k0pV<W%a<dZGk(e0LwyOQ=(|2nPm&4>p4D3Z2yUEhF&oiG?SUKN=qM8sVs(-@-%CC#W%1i$X~}-^EkQJ`Dwa%pGL6kS=j_zV(pg25Sm!D&O3oCNjOA|)w#C_$TJ?a*h{VI=aMW{VlV(Z^W{C|W)s`&4Hb3PibarZ@(^(Z%PGS{%M&`hDU6wrw;|t8(v5IwJ0W+i-fbUMugtrJtoiqXpf3f}H&ENMc`e1u+(HQ~kzCt1J(vt$diuW!!X<8pNZ5>tiB*Wx#hoKL1Vx<Xx8WPdBsx!#yhr^@vm}<Q>W!GJVl}(~zq<nLks{D#k@}Qi<3qHDtj}ga_BwZ<nAMIKUVKBzb5%%M8XQ+6?W;%fmahe3{jOQsh3!SrZFGLVWcZ_6hEk1l|eWyzHM5C+8B<v1@5tkgWIX`WH^s_S?j$i_(<_@S+B)BL&ia$<dSz!O=S?dMUprYTPsi3|?qvPQ~Hv$_xR@%`p)~Dk+q#O~X3j_t=RaznwxNN9RUPX%N5~8+hb^l-mhp4M%A*efeyUmA}*qc-^8s)mGdecL~x$X=xGz~;~n7CQ3Uhw*g#+aMcIWaaz9V+1@m`znw0p4<6aHH{fB4-R3Zt$tK^X9dhe9qs=4qP9EBFXOT@3Oo6kDqwc(7j1E6}+ePcmD&aw?R+Rc<4)RSHo2kIE%u#;=Jolo%*^aGkm#KCkdHBZ8g9E`on>9Jd1E%t!xEt+P1@Z>i$Kj4Lh&;x%0+s!`zi8Dlm(nR-EPZ{@IL14}KS3c0<CseregxBbA#UcHzytvCMpOgS;RD(Ycb3w&gF%^Wbl%Rw>&Hyqgb`;l?e7+@2>lbm2m^i4&Vj+RU9Hlc*p<k}&^>Xw<Y;0x1q#q&>i=C+0ZW`^S^uoc11i>hc*uL{-Q#clO?HKes|6PgVYf?d3{z?BxfEz5I|p*@oxgWLHTgyZI*{!FpwT$vxszzWA^&6aBHK4-(#g_Ug!Zkh`-OdcmqhEr=vC8YJ35b=^PqA{DuJ$M~3yaY09o>O-Y!UM3o46mjlJZHA*KIhlkL;0q18$lgznb0SU@%+ut5FXL#ox4e&f&&xX)FVbBr|L7z=x-4@%S$b+0kK~E9g!yz(+RJH1Rvk<T-YAVKi97HPUC=8opkvxS5h_XJh-xGDZ;rH*YMsh%^gx-Nc6-ztxNLo)_mm5mwq=FT)VCxrE{grvYb!Y($5WEpCRHGM?(qqG*6E{aa=s5(`>12<B-0b;FI!BHHsXXlPYRJTITxa=>Dz|oez+kl7Zw@FTA0N$XCSSTUi*?1`9WeGGJB#;pkfZatmVd-^%kmg_=@Cb+=p^lmwKN156#)IMGvBUk#!<|wPVgsAYp^wEwlSFS&u5W5OJ`Q*_SDKVtn`NgmykKHuJyVD{`dq!smHNjT=C%T?^{k_yC=K2^;XUSDb!Xnk>KO%<qs3sW3zk5me=sBdaX?u_PG}TOH?yBFHPv$x|tL$jIB2`|EQ5SGhmGrtXF0B))bDI5H4T0%8Tbq2w(tdPOwKQ=P|YnaK(r=RlON={Se2xUQArc*9s!ieM%~&`uD%Hb1Zcb3Rw(G>2d@N0#OsSv+$&dA<B3cXg1v+KszM_%yw^vM+wMFXlJpX6skER)GywZd~PW{F=XU&2jO!KP>)sse?Bk6mL?g(Bc+rkhFd=@qF#eb=};k%Z&!zz&fvF68F4HHd!ydu4f-N%;Ou+i?>d9zh&Uz*PPe>kkyRMqGn^1ArgT}L4@W6$qPELR<Y8!V&*p5hX(OAcc?n*AW5`x%`~geiz}#}ax1HVAu1$QJk=IR8*%s8RuC<pQ(s(i-0j>B*y-p1nh!gL?XOjxCviwa=yb9hpv2I8*pclI<taR?%!cY0iK!_XxC7FCV9r$HU?KLVWAD|`<DJgYAqIPN2!_qY6ZtUO(@;uItf}&sy!lA(AD5w&zm21>KVmo9X1+z(fJ<Uk{X|Rq5MUwf2(plLsvp;nvaY1_^hkQz)x1<i6p<!<Tw#4MZ9{{r7$wj;&D=GLE(yjWG7s4^0rA*K&BVNUTICE(Upm|8nZ?8@%73$&Y8he#K$$yL^^w;>ty+D(>X#kJyI5LR_o9#5;*2(w{V#25GqoEK0mWg3)$vYvJXGFkWoDs|_*BP21SlVPZ0iKgYec~rNpvNoU!t?%&8(Ip`a|4Vq+1kV4f6Swg#=3_RFUv^mZRvR>~v%F^<0?X98~rRTyS2XY1P{F8ri9I@mVX+P+@WggEr~-)ypO>uH@p1FMgGaU-@D#7xP7;`ZY1N6{l;3-xfn>7n}!Q^rS<h%eu_%cSNDfSF&L}PflMXs+{}fujoT4Z<?O#3I<rVlv7nMU^>l|3s`u%{d)Q7_U<vHILyi>&lHV7-<r#<Up@BDdSy-U8THw-?bm6xakhQw!T0Ox;QKT$<JPC%FFf=9JDhm`UCz5V8tL87y1)BL_g`_&{WqU-|NAwk*Ru=ZUt{ykTN%rAcG+HOkJ<JXI$Qz_=6rD=kMX!9$`-@n92PXB9%x`37EevCqRCss%wthwO}4gc&~_WVXj7}@@iG}wpIYtzdbhW)1HY6zPCVj?ivJoWdQ*;rh>t0bLk?EH%C-~Kj4ULD7A(<-5wm3pES>bJDjJp(ssilRAc0hn>flzV=G|bM&6=WeO!k~#HJ-7ibFQQk62540;A$Vt_q-^EiS&gofg)3h>n{6w`tCm01yb^m$si}u$Rj4aPXpC1Ocs|IVLW)^&HK%asZ}Koao3Mk-@jN<U&cE@jr_%G(BMc0o|QpGAY;AdoP+S3!I(?7l8PaBK@hBwaQ39A7PfAb`l*J~VgNp-9tuRdUT<efb8eE3i}RBcD3o`Ec?HDb?NSks{Q;T943(BSMM<8_LJz{VChfee7<c$&`_$Nd7j3LxqK##gG4DTKpe(T`=gihcRVXy75b6PomW|>8PA~|Dkx|o=vrSIFA{J0l)d%a-DP*d`%@sery|PjIcINhq?|rL0YzKc9MLCgw|6hAw+SW#rt@+)*A}{^8D$_`^QWA@WaSzxAdkQZD)V1%>v_?w`1lp{WkSMq3x6gUciHMBIObNVHPv7dQr`sqqV~-POi?fK~XgfN$xZocY;c6Mnx`MY`!Ow2}Fn%_(5YrUbx&{5+Y@B&}>slx(CX)4JmdkJ+8_BN1ERV2#Ds)~AIe-;y9_NoIqgK@T<e$7g`nB=m^?&*oQ)1{!IN*&&v>12&;>DC5k7wf`2v%Bl3;(mr6qNn5wV<64b&18Y2QwSo&(ys(LSs(Z=UrrdrB%8bwQ!TKJW*r&o+jUNcF7G+_Od9uA&f7(24NE7eXMN0uKMP!&pZM<om&Vu1*BJeb=BR$kEisj4@kYbO7-gSF^m`&J}8M#wTkFY$?)LmTFSl%Npj40@o7_KcHE|O4TP`I5OFwe;V;WKGwoTDe}wl_M>wOfS|SkVi@O>oD<aIQ8^p6AXD87D@6tjLmd=BJ1|C_8EVh3jvREztZm$x5w=?4JR|vo>(c}*hf`5=0>;UiAi^AU`41be2`~w8yOf0y39GDWWQrh=;3NzzT*C1n!fj!Ft8K$86CV|C7eG`_k_t}4aSn!sZZRd#O$`U9Z_lbcNvs>39`%wP0m06E62KtFD13!O_+3aCjO8Vs3zaDV1t3zcR%YmW}2rnYS*kKZ_q6@=8gsM3AIYAHqLl9Yf<|WUc4Lc^5Fl-Un?X}tJ5G#$vH7SdFL$wHE^cH=Z^=_K|h^=H)d@>es#FS7dAtYatdFC)ji$@X_ygBV5M3~2~Gpt${h#4_y{;GU+*zWrIyOYy>2nxv)->$kPPMi{fV>#_zD}&BtJ%TR6fm|mBp4Pf2i*yRHGpRG=u*k_XqKoii`^rt*`RX4cdQp&+(U&`;DQETIJv#IjUY~%S5q;9}98x{ZY_0YVQ@!o**`$(B!5451nTyrT{>o-_6Z@)(eK)So2P@nO!X3Ntqt_OpbP;I9SzQh`OGB!smutS4S|lo|R%yP!vt%1S;KY3J4-<>zAsgQVA<?&I7GDJM0>{)Fk1%Xymb<9q?2RV=5ccpN`~}#c4~GHZT7~*%a3@+k>o=L$wno!SZwT|P)@TNUy;vAKw)>4GLi!n<^WjO5<8a0-hIPi(@dn;dKR+C57?lV!Jm_S-7rfl~<wR0gKng$9!4awR9IoX?x@P(g_4?#D-91nIk!MqC)zA&pekZhtH?(D@Rc#T#Hv5A8sHzy{t4AHyIn8oq&HURMaqZ+-^|!_J;xIAd*A~~t{Q2+)**{Z5b?^A3C~*X^)k16F+`=Ly&kTYh{A!B7h#mB=L*5>rN)^1y#%9sP(+}!KDnN2Z5@)!K!ipZUAx6aBfQ4i?PY&KlEqK3_yc72O+*Psr>AkWSEO4JZHdKX7AUPI3F&b5I{OTkZQ8mdAbasI&UHf9U=FLZwPJEu^lEZ{eJ~;%ipxSKzDF)DX6Bp+Q{!2n};-3lxzaWfGgi;(vXr6c{!Z!v~5JPt)@)|=BBFTIqQ-uMfEciYrpYu5RTw8aH+;g;ene=mkvAM<{m_<WVRq{Sq3({=2XE?P84`D41o0e0@NwPf;Nig4DBOWR2Gt-&(bebhQ#~#$Fi1{UvW8xq&AJ^kC9Ez{fDLwWsI0!MQ-vv01$B;b5Lj1#Ma?KtZeArvJOazLu=de9wXi$1%-@qTjG&0p~&v2DHokvL5FlPhxI0?Dt*wwS@vB{l^Q><3U5OV26i^P3}jW4ii&B<+Ge3+HH2>nC5)FS*eky+!Lo^dH<(tUXH_M=z5%l>vDT7Kw4Y46>@uG9ons&pK7mNy`)Wwqk@e0p|R)6_}3tjVroGwhuWa=ZTM?Iu&%Sv___<J=v)dpJi#9D-Dk63&Q{-y3*~#BVj*8%fm~?j&L@j$2Mc?_a-s{*u~yQVBPA@>UF#&0r9WP#Y!Fim2#)dF_70PN<PS@@nNLPU}cSGGufxn%32E=h^TvP^UiK?EZ(wfdrAFUl?<hetqP3oBRv!AZg2XC_SPw4I;o5gyDKCa4lT}oCwm&uED=gRgBsDAUnU4qj#q#Y#m?T$%V9TZ$a~t-3D40jP4TwW<djlFR$H$*B?LFE4L9=cHDMBi=|0lA`+P0p|gG_doD3rl@APJLD^l2tf_m$@@erC0)H|&WU6?S3`!hz-On_O>ZW8`;K8l4RJ9kc6pDBT#IRk&4O*n^jT^^MY$={haTtlHtcLpX&E7{XB<pwM){61V)!&L?ae!?MisaM#4JxB#M-$KddyRF~QiHl?=OPn;#1&17I!Hv!gg_s35(Z097HEzF2w8Iy{v?8r-AyF&Re^zhV}lSMVnhtNRiJYz506c_m2roy*n6^BD$$`VcvIT5X{9v2C17J`A@6!Pq5=x@?WwyY`nk{44-uxC#W*7R!Wn=#qhQR8NrnRKc_A`Fht&=Gr$*;I+#ntx+@f<HZqvI;xr~3dL=NK4srlBZG1z*tjXB(e+m(gqYwx92j07v?QjO}iVvbN-E<J&zWDEpD58>9sATtc&y3kQrrjj9YQin)59U_f%Si?K4L%3D5&y=|wwxlNlrE#-;z3nH=wf|E)INR1)fO0BLj0~0M-l5{NQK^5xHz8A<<RSPc{)CO?u)99kkDjo^8fXQMvY6-jo-)+cyYVF#@jc&ni|xC`a^}QjW{-2J=@pUw2vYgme)iiw#_XbMHOBIJBD?X6t#c?dX+QO={dAdqQ%2!-J~wjlUiU&k_6DQ*)!0Fh14Lgz1_gaV<YEPUg?T2O;J@nm)zu=1`8LOa&BTX=dYuzO(r9LVZL5%i&EuEvj$aV#&BlTdMf6c;NA+0vRWKCONwe2yS7F82DZAJx;%P7snF9U8z0+6wbYxB23zg)vIfXG`hiOi$$q|;|5u!YYVM$9|4C~lp63-yIfoXt<10My(C7Fr5KelHfk#Ag=dUe4aTOn=~b1VKX!k@8e>qHMS)6I*6*Kdh#nT|ADte9v~HBp6Ld?{RQlel_7Fe5AR(lv|L?a$&%!h5aWw5%7tfTvJn*-ZZj&tH4h%v%CMYfRTH+aaGYTM{$xCxQ7?YCOLNuWWI+mF0N58xle$>~_<n_t2x!N<Z05Sk$`re2Z9z;2s7;G%>!~c3cdtjMNepRe+s#K>@qC_|Y@t7ROtspsFYc&L(@qF3j4DPTsuPbNne&-=2ogq&pRE;I7Hg(v^LkeXSFm2^*D4v5m{7<RP>lD)vLQVC8~DuJ(M7b>6TNtoN0_I{u-(0v0SmXxyd^E_JJw5^gQ|AMf)+8pA{C`XTlBu==GyZ-}xyBx3WBkkrF1dh>9bgE)r#zcRbN=lDa^y|vx$n%|LhF~~M=9z*xB;yzZ-p$1(uiR(??dXu(Zvep-Ja@Bv$e*BTPBfQUcth?K>?rz7ryB+I4VmrR9JVKQ}z!3^J@7y%{e|pP4W6vHwK`Aa8VjIC>{=Ub`Jce%F(5)M~b;Ez-Bl&~3_xC-f<}q~ZhHl-^tsDNHN4)w6IO3JsSHplH<rD;bW)OH7qI8Jj;ReIS=D+f>H;)y!ZpE!zaqCw8jZwxw0bKrd(TI7hxOFRT-HKbc@_VA7+8+P~RVzOLrW6fIL)2#msfQtIgs2ie0i0FJe^j(&9;<HMs#~|})~)^<qqP4|;P=NwbLO$?)~&jAt8U%u?};Ke{s1VlR{ot}RMFftgnniaeOTevQKwR6xUKz(HnV8cJl5R0HMeffty}vyM!Ek&;QS{VPNQM-Saa*v+`2WlZteF(;hTR16u$jWgn7kdq#^z@gZ#q^>Qqpt!YbCv_Md3|$D(!fxZ&2_aO-ZkbvOQ<Ny5MO@%WQW{75ok9yi>&8*beVx9-O8Nn*DC03>E(^Z)Z1Qan}q1byZs^ss_DRn)2Sok6woC)xSeAv@-A)2+Md*4=dLZvGpStpA3`?N2iHuR_Mm<EC48)2+Md*4_L)N#OP$fCO&V{sqsZ;vv%~@-rXFhgH<6qE40Xz^a>nlEps;Su~GZZrv@n?v`73>))89{y~oKpJeurKxWP3mRonrt-Iyc-THnK4L6JV$hFIk#adf?fl2NvwyM@akBu*z_@z!SC!^WLrBzGinmK*5r;?MDIUF-gz$L{t{0`ZkJJY20AY~6lsUnT(WZ&$C$Qe+TNp?Zyw}N$Y%w83i^Qlf>$9CW{Vdbc;=ZExVk_$_@JkBpieU*ia#x_*O=nH9|TX?Tb^4aIHsbl}-8{{9nK0Y~B7W5`x*Ql$)9Skqn@sM3^>&D@7dP%5~MuEIfP6{HKsCE;^`ptVW#}8~@isATx{R<1);$4i%#WR3|VYiy?Yj52+X^KndxOod7Q7>AIN3kSzGpYY1_0h0X@{Za20mk`}3aFumGiw!PCgP${_s-&R!Mg|pnf2KP0+x>h&yUB56LL|mBYn~&;#niITuH6yK1TpgG8b_H&ld}RYvv}klrquB>WiVjUQ0OC-ow`;iP)i?JuXz7%!@D+`f&zNY~R07agv@r^J?sZQ`=GQfM$d`s~WF6_L0LtUMlwn;{>W)D-*vG#|7l4pWq^<<Eo(BpS*A*KuQsQL&JGS!$pKc`dO!+83#CH{x5D7%d;vfPs*r2ntT2*iaOdk)_Bnb`3=uC=J=+tnqL<%)A0~-!e%1!fu9b^DJvD&$q{+mWAQK@8b7Qg_rds5y>w$Nd7<O3g<XNa7X4K*mW<n=6Tec$=)9P4u>EoC9A_f+jsx)HbSMWY0Z0HN0P!LWGQkUHoNk9*k7~5*8?tnCm@6u707o8>)1ZP4tVo-}f%a8%<I;RT=B&r_y?DtYq`lS<*NKBYi2E9!4FW_su@&f_ix~e712;$os_gO!<H{(HIT)-Vk<%R9k&R-39I#X!r%(m$qbHn@oQM~QS~f#bTD*eF7E5J)pQB0BOYzNeg8XvgS@Adz5*Hd1YL*?8*FD5E0NDJUxv+8>!JIDQq$S7YhW9?Hf57$ANi|YwZ6*j5!n<S#k@9^NlhBNRo&E$m5Lqk{?R4=>BAhg{JW#ICC5hHpB03PYr;X{tPR&?qZPvU-!8?11fV2-oovINWMoeeDF{@VK926+7Bk7e{6f2#h4%u;3d1d80Zw^6!qRCWwZZQK8gF7Hnx>+kq7zxD9Gu(*Dh^T2fM^9By4+t7@8VNYE%_QTX)gJ<@QOidaZ#*K7<CZ#%rz#~T@Z7YhF!~5y@Vyq3cEwdNDQ(k;FKM9Ef`7@uWQP6Q97v5gYM$CKLOkkG;@oi#)||vDEZ@+;nB-h(XGduSz<Q1&6e0B9KmLJ4Q;62_l-o{<ty{$c6$|ObukYW#J3?5aMOZl~+yBk_{Uz|BS+7!S*1t|*qCbNtT1f&(lr#|0htXi%kER5Ai+a6vJB{MfQH5TENonv(2@IV*1nz^Zb@s4|U$td2jF*Np6$*MDJE5Qh#plwg$pWFg$UrI;r)AE7sGZ#B2QS7NLCmYh$*%_ouRnTgxoaUs-o1Z$y!ZBnxAxCo7VdrT_%O35cH$r)$@TR)<8qBBKP&ROBS>-A3wP4d-NeH^aRsfjD=F4evUTG03Z?gz<WRTyc`Y7jHeZVNI0wqKcQHe@6Ji1vdphnINg(q{Z#s=6J1g<{PHJ3{LO3P0$*~G9Hb5rm#lDGNW?Iowf)*@HJ`GyQw~aC3g$3uk+Ru%das2KBVoGqlfOQTX^FQgQ&_Ak+&^QvDrpviXf$b19mHl!Ng(aU0TYMK6rgDX(xi8INNi3hl4_9aeobh$!jV)>oLZ6d`n757grPt|4S$AKZ)A)W)zc!!0$Yyh=aj<^JgZ`v!=+bb_e|t)+j7-FNx!iL&MhT&6am-jjsga4k`p;%wQyaL@X{+VPtJpZ1taltMY&pxj_;1Pb+65Yo;f;x!O3NY~lVWHW`^?WSbU0flKPW8-j_a<sy{f@0OdYNMVL9R@Z12ZhwpXI*ENB)B&GuG=eMoV>%kp35WWe<FMlf58Fx+<31mbY*RBL8GtWYadi?Al@AqGn-IO%^7EFf6Fmaae2nrsKlvT7~_=c%-s7?+z?laL9KG%Ext2N_rK6EYE_b@WVcUm*Zme#u%bfvwx4{%kOeIk*LVDwVwT%EpEeyo#&4LTtZU5(y{{L`yWEjHZk<2lm2$)s!Wm52kF%4~RvQH}ks%n~yBmEaIiv0>J}0nOAxDIS|t9pDCd5K(eIaci{MPdAxfwbqj^!7qJQ$^wDs1S1s!l&>}zb)LIfiIkU{=^73XbT-_q!>OKN&7K*D|tmf)$GObCsFFDtRpzDBDB4a5wtiWPh!DYYdF7@Ov9ZIXFLZ@5o5_eAAV-NW;AT}-}7`q9hBq2BY&08YUN*61U*L8Wt>7Y$4HuGTUQk9Z4X^I0}q6(AwF4wW$BainmUouSU9E%_cWH&_YMIe}C1os75{(GfXClR{Y)<H%kf+oi9Nt{u|q(om6aTu}8Ui1*wI~{r0iQsu$kZ6}`CZ?7OMf@pTT38}`x)luTa%ET+Xb1?BIMN-5s9`yrkZ(3njX)8C)rfGha3j?a5{>zNCmNGP6Hzy>_TNZoN^d&pjadevDym!Ne}~ax%I~1O9YOjZ(QFtj1|wpJn=Smyf`LxDs7mul7^UBOgXS&GqSwTM3`^DGmN#RugTqxN7pgiS632(z#!-Crf=)2TjIfu)0nUckJW7?~QVn>`+9M-(9PX>Ig&h4SmF<Qzp&;K47`sm)|5)FpI)-9P{@<Ik!+o6qL-PH49rfzIsX`C;Rj$ySJZy+!jh?-0gNQmv9lGlcXZ`+oI!UE?#8@4kLeVwN2#D(ADK#1(LBUR50Q?Cjfhl^1c__H4KvHbPWJO{P4)x9XiQJG+PF#33p-f0Qojw_C>QbJDLr2qpKpNEY2Gn~)j@(ZJV?GIQ5DZOnY_bdysLU8t<R==E8>aq7hO;{Bc``I>WVFh3X3kWO6r|udSFD;IDUD!NCXFv&4>b1nV<D+8wVpRRu`D;9N;;ZKIm*YGQM#Ln3)o(14LOP)GA-5Z1e$_Pm?H4?5+X@STzRDpP{cn-#Eh#1b*0iveNy?Q=(t(Xl!D+i2w&we@sY}%@hL2LZoOrk3=O=b=2yBH&ou8sRIh|cdKu0M$}x*2zEXTdgqZUA`~xN!?c^~B9Ne}ab&2RqMgtdB#?i%qV5%S?`$dmnu&(-ueNY#@OwCo~EFE@R?a;8J5)Z_jDGLE%&9(4l^Cpsjb(%%%5T9|4)TbezOkNlXaLE+_4_8K!M3d*T738DnbS(9gfE1G^mS(wMb#JS97rC?U3e&E;Pi|)mqEhB2X9J}2HTjJB##Y4@=38Scz^=BcyNKTtB_jam<I_W5NY=o?L#dF&O;6=GxL0NaFgDa*gTW`m+Q<th?ckOCnZN`_&|<-6HHd7~?weDOn1WOWxaA4pQc0Yy+$>J%Bu?oRPPAJm^WNdnuee!gvMx*5!hI)uL7Jc3R~%(vUmegXEDoAiqlt)knQT+6(x>IoS&Is*D95tQQkkqYX)Q>Li_-X)5fpsDSU?zlU@q0+RLzSs%Q%T;mgZeqns4Swe!iJb?d*~^mHvR-;pd9nGMO|~Pjb^xJ#o@dImu}NJNHN)m6HgSb672G5e+09t4>?IZIN(VmJ|(Fd&5ya&8N+pDvV9TItPVZ`O#aR2J6PoaplNxPIP`c5VuMZJ?!5KDFpOGeqam*NRL0wIMqRqvFjj-8Q$#;n|)EKYg8Xc(wtASNPIc#cA+DEtWkB4pJ_PUG?|q+g<ClTU@MQwe)gEy1@p5q2|uLLE=i6b!u$EjnX+X$mCuhnPdw}IJ4JI=DSpd2x+9a(Rc=0{RM(uyX<q)mW>toybkj%!H4m5O5&H7&B#+~OuDDE51eacR)F!ajty5wKVK=S1Un%tS-N>%@Vej~T<MoS=;8y{0pnd#FXy1ISpMOGB^f}O)F#2I-D@W@4b^l1G(M)vauKt&X%PL?hs*ED|jj@aP73ourUBr?r7g)@Om2F$sR>ckEMgE<OwM3y!el_pqt6U9y!*rv#Kqol8oAN16_;HsJ-|+w@R=8Y!MaHId+T=iVpe4!PK}#UoR981UYdu0>Y3OEYsq4f=_PaoTu40!eKlUm~hDdA>gTW7Az_SWkpSq)+3$-dU%wxvQS{pIM0?98*!}3UyD(R@`1mxh0)=`=B(Mk;X`9%uE_&B{dFj;q19A3meW|#|MVZgDDL(x)MF?W?|BH&cSFlA@R3zu}^zc@%PDD$fOJiDs89~L&7cDWtJuwi5iJM;jJ52w;l+;VR;G%USTh0I}D>8UO?|DKrY6_^E&2!-Bia8pK%I~p3Y4GMUaS-lDFT}hOLJ%W;Vio#+Bdo_!vs%H@=7;>8sd7>h*|3yMo7tu61DMU$yd|H|-(e)7vJ7lpZ*wOS@R?M-G?M<YkIOl9uoTSG;4wk#x0D<}%;+El?pw=Ez-jxRRLIjcyO3~B(<X>cbKTb=J)@GyJaVDAp0*swJmj&!*iRRFtMa^4K3`<`jsIhr#yEHgzkVkgE1;$-0VESx+0sRS8mg5f60JOZBtn~M<kCi2w&(W^aeeM#naE1~=X6N1UaCSg@!SsOAgHV!Zx1|sra#9#XXZVgZ#tYok81t!0(U7C`%iLNPn0eg=RZtk4=(%cmtku~tyjOW0{`Rv{V{^VS7C@7{g>+DVizXvMCPvF)>`p{K9k(HfioV-ux>aT=G_Cp+4A6=$dn%KUQNdswma5Co=&LbT6QsNYtG5L2!2BAf+TB^d&&qV&rQvyp)3R{A`0*5t$<K8g*{J;1H)oZvQ7l)oDnf3VKT=15t?O(rRd$-p*<F^{R1<+*Oy14Zf7O62zCD}Bd-Xu>m1sOqkR?99#%Ud}8ely6@2&d#oBBKdVpX@bKTnByR$1b&3ZmapBZCN$V#2d{`Vq4k<y`g4_oJDA5uVHFI&lh>+7gMn*lFP`hT#U)BHoc517-n?sNSBQYAjH<a+Rs8EFs-AHTFqCH8xDU1O^y&yKt$HG!Gd`N7Qc!#(zJNhlcxg)??DaO#Z~mZ6XfUU#K3JC?|6M<7B9D^eIoWS}8nf)uFobNTKIE(JEbJwVajGKz@(dWegy$!_7A5m70O@mVn!ZM;?uvpJuq?QyeWa?-A#V6lK871Q^raxX->)w;%%E>o<EZJpbU$o;U7|qadd*wjF4&K~l;_G{<1fqLyCA^3s&iY5Paf!E+)F9$}G5w^E6MD^<j*?v1Alr1j&AQbRdJqC?@hz$q-Mfuc=rmJB;TM&0tu>duD3!>dG_oQOmv#qe6Q4tQk$$tH;4My;j|4wRo02Bb|7lYARui69doJklKA@_OQgc%2ayz%4J5DEX3d!{qv$_c~GWZIYrxjk%(xVdX~2T*{s#x#SK{T${XR7haHuC%p9fA{=y@t=4u_syXIWFbL+Wq-TgnaxDVF9TQUN^Sy$eGF2&-*=v?)Hj?%=hmzM&vY{doud$$4=&P<%z$J+U(G1{`@K21z#vte;Z3ZrAqCQ*3(~@avh9vUax@=cR7mUPCjX_!Y9aa7w)?7_`)SC4Y-zjU@{{P(uE`-*R4o#fc45y9&8__R1W#;=DXuwwof+<UZv)0)dA%*6~?Q&va$sC+XsF!5Hja80LWM-B0$~Yjdbz~wlvpXPXmTdQ@w__4kxOBbFX|pn>k8Dm&h|znf4s1beajaJ#Ri#`Xl>_CYk{Db0sQTDgY!f0Q6katZeDu{dz|1TO^hE7)*2;YOFtkP3vSEDG!>FwoMzWo-|Lc^QBCD5y)h=5}`6@vn`{?P0ShDUY_F2-JGWMtyaa`5GNvuTtZZs#j>O&P}Cx%RnBHD9U)IPfF@|-OdiLNN!KFEo9>I9ZS1>ubLU9DAgYun+IifNQ@k>onyE*`aiAxtX$Ww*uk_5MXh_pIlJvNXdSGd_$JUundSUYeV&x?1V4SRx+{aGGo_DFw^dj1|dWzhe1)q?RzrDE#vH$h89LU%2WKW9FGn@tJ`&=%*kTMV)%$!OAYXB=rdmVphXPVo<mb$HwGMs-?P2fKoGylPGdro0&#*N3)4VTcN({v0{}Nh^D~AE;3jpZ5v876A&mZyub6`TlM!h^><ExX%qN6{@ywkCN3Aw+KP4Jn`bFQl$53C=%&<C%8rNSou#!h=J?JoF;~c<0Ozd|tI;sWdoN|C;(EAz5m@+-EHJQ=C)JY$5qEZK*$*@MHBcfO-is5z7CS9%J+-s6-}YG5;PX&G{_RtZFfef+sVgYwx*9g^eNaAM3I}`}rI4wW!Hm-ei$+%;ge3#9hN5g_@b{@2$ko+S3#%&-zDrm;p0F{>jyVY$x>;wc#-Ok|+v>^XL8egy=P+8iipAIZuRwhS(RKAERBvw0n~Hk#8@<8b`A+WPnUo<W)Bj0sn50a>KGl4Zb=(pz7=v9Cw809Qj(2v2jN3YmG)X4w+k6R&NnUv|tUHM($6uuddVl-|-Y@vcaso3`lD3|0W`EWl^oFU9*&(YNZZ2qCI^(kvK6+HXHK8?avwEE%@v_;!fxhU~n_rJAUg=f2@v2OaQhF6OUWGK23bS-d@h+?ywu_1DiY<xfvL{b@xryHJmf-I#vEMDne%!xeKL;Ri%Yr<w(TlO%<K9lA290Fk$Mb?$CeEY0sFiEwnfDL0{SVwmvY*O~-4C|;_MOGHbt<{>yj;fr-L8mhr^7<SI6u3(LKF}9?b51M`abrVx~5~FDioSRmp%Z23$19hVmDe@(rDFgw3=!3i>=;iunyH<#yEWSVFgMQ4)>>@7<Z^x1*i)o>1qU2FfiyS>UECs)hS>$9H{0DS{d^Y+7A``0gj`Kj=OBO^Jacmswa2H3fcWBWVfJ@UGTd!9gB|3t@?ef4Ytsk;%}DkmEc>4+|gTb5>LZt&TDLgTK$|2Vo~y$WfAF`;yFx=Aa~qN>15?~KJ`legz1MxguaFXiz5-bb@$G*L|yy+m)vE^j6Sgy>_pvY0?audO|A=C=PeVma(5~Y%Z}dRvogKzA+x8~kEXGZ<PV4D?pPzrIf!$o#G>)z%+<79#5GBTm`?j*c?cbaQS7UWzU1~>nZwkPxOk%&J`CI~O`R-j5{MNd*C;_^zhG%>$DMpD3A5P7U1xorL@!Zh?kI!wi^z|soqC;+c+`N*`uR(6m%|`m^~4a;!W}$3_2Lj11A?*DWVcBt)oL?==<+kmg!k*A3AG&OC)h@^?QmIe^Dc5e*Ge1RAi0rZoV~T9ubmPtHp9AoVw#CZiUjCUF$-GJbRH2W{`_<g+T2bDQFVr+c|mP;55|sARQVb@1MBAF&MT^W-A}EQ0NG5EZ((Rl2peKghSOlVwXmfuzo{Wyn$nklcEhB7ac+4*JK0c6v}J1?t!g;cv8~J_Vq4{+&ZJ1it)b#wrIi0*0ofKnc@4?t<b*)VF(WA0iG+hB-Ev}6;NIki8wJy;HUU|3=dUlcp@J5*_Gc16@3NN&INT4vr31tilo+2-?q)fTLzP}sEDPTlKBjT`%aZ%@9IsC{`H?G1KhNSP$iKGCg4SJtb@vrg!Q-6`+?ye1(L$ap9Cm4r2Iq^Jwnm+Du}NpF*LUo}Cy{cV)d%jo?811%zvY@F2`g1FmmP_9i=XO{82H%~KiRD5?SKB~8wA~g^*87j^m$G1`QFiySd-&AFjmU;M9pp~wMEZf4(@qfp{>csozIfHtjr642gYYB!^EE(7f{NdoBn}*^dj8Zam-5Kt4yyWjZ{~8JBuL;hd^L8dn9J@Y+j0+4Ze7Y7Sl;n6n5~VigJ9Uo34lY`+cx;Vd`DJhdO42AeGHjpr8vTP1Fb-TdMMm3SmZKbx9|JCl7aUWgF6(M9u3RhX^n8;VvgW+<eN#-2>-_Nh`W8!s*(=gDc39gi%`%`=fa`9dAzKtyG}3^x(J}VGM<D6h<Q0DFS`@8e9F90V1$G{OV#~Hf=UWUH_!Ox05pton|@y1zE>`+q^>Kt@DhLd`wb5GS<5{>uq3#u&(bmpQ6uDqaQh#K%jglYY-`JAi*m+o21&RxNTLwr>&~nR`q+@s<~~|zNf7Xx2=uuX=~GMYx8^B+H%|4`o^}11~g~=soU80H#MeCE}hQxT}^-!lDnJ-|Dt@A<7rWJd4%Kb*dTSw%Y?)4DMLdQ{zh2}fB+D)OQ+fRt?~UP^CNU#=56$eKl=f92-n$arPu-!G-W2T%`U@Lf@(ZJrK4Qkl;~O#zeZUtEZ~Dc|Ay5{z+oFVYut)*D2`_><x@->)$b!3+GURp<76h`b{Ii7VN2Vw^*|oSf^s~*6PNIQb2e;W-jg8nBKVFA(okz9k{i`_TS9Ztk9}i?l4v%RVuJ%{v0lUA5ne->cI^sMqXu@=!n$wwwMid!wQeh?81vCa`sIT5PhzFglcd7_k{Y2~V>^K_8P+g=T*I7!(WmK1Tw$ajJ?+cmzJW@0&pQ>s8;{~1@6|$^%Z^(rimP&h;_zx*mM>s#ntc_tbaB7nn`oucoccL#WJBdghOjV+F+<uS<Pi>mj`kw6CtAZl%*A<sBwK$8O*Xa{RFcEPv4Z1D>}(YrA?U}hgcwq0rBtnmHlDG)x<#7?Y>eX5r*&$Z2@*-T9L&e;B(-`*<=Bmr2G2vG=N(}Lctrg8CC753?Z(J3K!FuusOXOr)2yS1Qu72Q{u7wJwgMI$0Rz6jpf~QIP}d_g6bG4@p-Yb%<EOj!ith~M&!OUU73=#`I;5(ZsQ{+lLk7&c`o;6RC~H1U1I$T*J4J_r%#zu)X<EsXiX^}>h}z&z4k6xvLm6JVi&ejO4}M!N5HtPAAzB*r$(N{w#b3m+!Hv|p@c0S?C^smuEGdR7b;ox>M?N%x4^ulg94*<l`2Jxjmc*d2kuBCHFmDAe)Pj|lhJg?U8>$2OrJa;BO@wzeQ87(a&+TOwR#uRbgd-MYqyq1Vor=Z(ChoezGT+S)2lqm2ip|YADEffJ>VSOPad5?XmK8J@=UJOABKNK&2UiASyKEKm<3uTtoDtCssN3^ZsOd`WEdtXvCl`o#-4;CC?T<zi!+|`;8a$;TYGmEkgv%$a6-k_Bbfv{=cHYK6bmFKswUg)#+JvJIr3#wipF0*NZ>t!YhY-~6b*Oo0@V;-m?Wka!cc2~!x2Rq$wrw=s7Le7KwYip?{A2(1;XSkx#Aw~AmB=^;&$>Cs{y?959K#d!kPbjIYsFe&@auj95JZ3Wh<8rjAIp3CvzNA(KVcFKqr(Z}8V(Qm`2ha2tyn`$w>ecvQI4K3X6BPo3&FjCIw8(#n+H0=*S+EJR_W?uwoQ*mNpvB$KkG$SuvoxqV&!rA`t5&d5TXG+gOaz=m`<8w9odh#+go*e>c0Ug-wt3E2c~<7Zai9kZy-AT=Fr>rI8tq_tpwrxPyZ*@+04v}FKTxyDr(yI07V;u>sdyD>Rrnb*gmWseD+{Uz`R6KzA@}sLPxr=W8;wQ4C4P-h?Zjx8ysv{sT;!XH(_Q|vGcUGMXz&9ut=cm52@!vwR$Q=h}tCy!|$C@EgBY4ABA-ZjEih_h5CH&z~x2t@|2bYN`q&<RHNC@GqD*mAcJe#lTjQO6|V|f)$i9OJU=5Jl2cSjwS5c7Bxb!)<uaDy_ZC?3R$9TWm~(`<s5+LuVv@2XR7N*~LPEk1tKe-6c|vWB1#2_ub{0osTgF&$zbrg!dp`-;$T*=r?$QgsHOaN4x3=YA-yANSRLe}h4a`K}?HGhqqm|sZWfxxnP|hwNaC^7STkJaaXJ2m+&4jR-r7)8jwIDCkx8ya?3&-*K`-207G-iC1_<(7z0%MzXf)_qeU_h03^v`|Kd#eest>-+p&VxWqAtwrtqq%+NGm<Q*uy**XEE3JZwYvCG?dRTwWLk2u`7{51T@iW1x}`0?3s`!!PqcVAllXsZhx`5VH<9I==4wbA^RjUbog_iFGX%}K9KlqnBXXj$D%K|8u=Qo_Y?Dta7fC!Gs5RK~`cH^2Rt8U^D`}4FbaHUORx@Dg4lO2bWH#{`x-C;aCxoiuBfOg&=jMK=q#snYztU{8X8i9G-@E>rf1~rS82kF4me?bL9Mf)axk+;#6raWMSchX!%t5{{5Y@VIU~gIJ6&)q_CJM0$p=GibOJp2lX6C`FdE7+QvVO#umb|MDM5MSD&8*0Ep=M4R4bkN^L~-9}_-{{-3gEK_qo~m(fa&<cuy|vIjTF<8YuA~cn2O7rVQ%uGY^6_)Zp4HhfM%NBF@Iax0w)+&I7SEl+~vj3-(Tz>`ej3mNM^gBnGKlwC6VT2LF#g3V61kb%-<z0qGQZ(9ZiN&pH1y&)A0;sL`4}cK%Fp}uun#0JQ~^)#KmJ`W$a6_dBlS37z!vZ0~iFI-07s(nsNNlUaX*s-4@^lkUbaR5NIZdv^^R1OXwrt<<iB{ItUk0owuhFBfN1L^~cdfD_-w~!2;)E-WWT&Cp_I(LiO+#H<KzLw41!->&79CGT-G!-~N_n+}m1vS_gY4r(xsWTkS5I$A${#osMShOKM-`Pw0tVyE^E^r*gI=^^yJ66>pkoDumTmZ1<?&H^C20tS;3@Q8hDY#@EE}P(M3~7Yh4L_2PJ@a}g<97X!6$SrfyTey1pE-!_48ug&pdrk9y5BtD^9$zgB$L)Ys|gTcxBqt~a&{d_dh<E695#<p;H_lnO%(+%CZ`tB1U>H84jVC_c*dP4+w1+`L&b{0LPbgWHM)N`Pf#ohLBde$GFTi)wAhRR96Xyp^zgG=nV^ad8_{ItGnPure2bcQHEE3jj8Z;@*;g+=TojpEcrCfeV)G#!nv8@&!pXL^w9$EE&Al$V6iNxCHlEW!qiOG6R^0^NFId6n`F8Al)+Xnz0{)M$cxq-p!*MOXbdD|nH6amqcgB#qtV0Hod6=cYv0pzNXcIvxKfwYJeY#M$v^+*Snx_dX_PwOLbw-IS|DvqkIP$uEsl>F6>4zRP$9NS_t~c%)k4zfX75YT+ARh4`}DZ_LIyYrrEZc(_ht%O;Mu-3inPY2R;lvbVxR$eO0zrHxi|Vy6#JLY4zYE#cAkT+ZFRwl89Hf+bB3!+VE_wY$G5JAq$mG%Mo7!c^cUTvZ__R&3s!^-Vc20%K8Ww}|1`V9|{hXQm9RbtFlOJlDPOPB-+PI=}=HrXt6B&po_sLhEBq_}o2v#!KHgdUv=Z2+OCT_ax<BQ8~EPZPyS^IA;_Y#A)?M)7Y3fU^+eE9t>v)HMbn8Mng(2gB+SUXNW0^8B%hhh~ZRt^&ot?cYO3;2D1u97+<Tf1s5Z*C|Y6j64Z$pW@U+=5H4gEM7BY=RW&70!C(ZFKwMr*whyY*n`*~QlduPgJ0!4=l_uhr^(slk1Xl3&n1q3|1Q*L}kQMEK?;&}}mLnIdNe33nPB!f2HpdbZ#GL*Z-2i>l34V%lf6YU#{UO8yk{hC%P|AY)SXsd&tSDO|yC6v0m^Ux(K&S!_h>;{#BC(LXF4zgK@@k8CnA5_XlTT3tZ$GSV;q9>r@tbO3(v=p3DbxmUXfItbGGk4MO<vd!9d4bg8uxw`lYLnr5JM)^nqnVZ^vyJ#_SGecnodPbQH3N9ynhh#XDgOyC)KXm<h;hiZ*AN`I^4@<O;FEEa1DlOP?&^OlX*u_F^~E2WLI(s`P6&qz?~%@aN{IP6Q|r5O>%J2Af00<4fzg}2Oi}=GN|d#gYgXqBP~l=OteeO5pKC0N+OwIm`;<GQuHe}Z4l%z9_eom_hGAz5e3)tQOBD$dws9nZ}tX~p%k&WpfXMY@>fT+&zYpfeyk2_e-kb*y)>?UYOV5k6=qekDYx?Wn}pBBt;$bfCBs-Pn5!MW!&C4+%#9W`pi&M}=2l*cFTR1?(#!n})FbRm8mjU}Gt7?St9s%qMuLuE><{?P+P|@xF}dT+tp3UrjYyg~v*TDs8tm86`cIy(&OjL4m76h>>PgvM-doYz%k#_sJ(H>!WZJts`D6ZA{-u)YWiqf}1HLASp1mGRa_OZ!nNH{&N31!4kU3Pc$k1iSQ!dI-a3J26)&U_sCM-%uU2h@rtB)#hKJk1`98uXuz-n0PC5a=7JzCx-E~Au%b)qwhO+@m6_^dc4bN!xkMMwb0nzbpcnfi(1!GHAvPMJ!X%AeM)n?|k{<OP9(&IOlfdh3tG@5J^pS)%<`XJJg}-Ey7j_mn0#a83VuqDB(5!Z;XivnS7K@!@uXG&+j464fQE#KzlRR2x*qb-VaX`O5kn!2{F%`(_OnE0XT6!=~vn4Hi8z!H(mLA<Jxpfwo9eSTKT885P9zdobk!KY3d_DY2zx2??l4%-05ZvH^#}+Okk2XK-5_(0|%*)eZUVS@Eu=CAQ=ik~x5-LwOO7i%+^R^_Cde&c{+%fiu-3qPjG=Ml+TOWj=;g^tQBGofyN4<?4LnJ=*1lH*4J;d#EacHQK>seJs*nz2Xh^ej~<qn=XHOQtHp9BL1=mZwp=NcoAQ6l0hiv{zS`~QYFh?<ACA#$DGaHn=WW)qrrH3tKV|KJs?o1Fgb3KN_7Mwo(h{d{wl?g_WE}cA_bODkxCv<xV(#qdWh-3lCAMBG~sZA<ss;lL!1f_vx@oU(uhnTtf#7cK#y-fB+oA`&eS2$-^y;=r1b~NcP<_+r#j+WR!h)7DRt8?O!a-|HArh?JVnkJ{M>OAAs5k5VNuQrjF9O-L&FPY;oOV0O*0aX67ti^3~eo)7^_SWun~SsDg_3j<KhxElW8%^4}D^8XcHS*;KDYIoP<)vTdV6IYuJ8kHW&+Aoy}%)Qj*^2U$Y)DJIDC=z%FY@wiw4DXIqTp9qYFK<d+x!^>#VVmGq|u!=98#Mrg}1c6X0M&zpRl)>#F*Q$ODy%Z;EMDh8wm6DfaL3~QCl^nQs$&4wr0Yn`N~j=iTncH9gqAv4G0I)RNimK?ZHhw0?1l7Vc(x-OFv#Z;+tv^Am9OzJaP(_9Br4mK=`@YLR^tqXT@lnqM=J*ys){Y#N`=}T>8scR7`t2rGYAi>KFtY@6*Kn;=&^DxrnTII{KJH}ndtvy)^w(wMjxH@&P+(F8vm2Z@)l0Z!?b&PQx+vn7^4Hvkh3>ClIgzEkh>S2yfdrw>B8IgIo@4nhk0{q4kab`|L(>=DwITR^W6$!INL})#<g;gnAam()KG*`_Q$u#%H&76DZJ?>LWr6mdf*l4=dl-NjuIB|<1IiDYM$`;<ym{4&xuf(#{>uh#~Ky`UZo37L~-Bnq6h+M6@q2t`9Wpze!6b$&x?SQ>nn(@@u0L$>y71qtpopSqq?y07C+*7&NRO&vq6L|WXYHF#WB-QIvcDzZl%<QTKU6(pVWojcnUHoCijaK?X>x+AlFULH|gs3}Png}eNoKj_p%)ysBAO!qugMuBc!M4IDCQS#z5W|hmh@(V)nnlE+$&G{R*GV_75?Hb4m$LWQM(`?sWRh_jRAVchYg&I$jIB=6m-Rcox5B_W0WLFS9G;)D!J+FWmo?`v*g^X+#7dB9$hKp(+p%r>GwoD^#5zJ7KxxD*Lo2ZTk!A9ltdkl<QUOC)QNEgslumLIekUZvn1rSSfB>9sdU3DSkJ5tsYU?0F=*0v`h4X|i6flRdBh-L&#Z(_}bF6bD1gI1GTZkX-7&)3MbE6!ylH&}6@jxlDE5e7CnL%);L69=`GSi8d@#!=gn+--7<P?{zxgLKy9~IyOnJ}v=#R4-j5WbMj6HJiK0b<%QgdQttIyS~C!ula*>|h+H_;}5j6Tt^pu7k3vcICu3FPey)rD5dY_1pIcnir*q=kJa<2eU=g5;v6u6-cfmTJ3@06P-rhACFYfTFJSj1HZ|YCyc?gefh1JVOoevEo;`@|KQ~1;d>9+MB%Ey*+pw<+qT|xoMm0CF#dFiPyofWmWP{7o5)?G6c_$07k;dI*JO>eA>NG7w$H7o!*l&&MWhLpiWs56wmGDXp~7e`K2N%3H7|*DqxcNNkvNZ3pq^OD%ykuCRhB3(sk_-ZCnPd0Fl|;Gt)r1!oqf{a*+*VWRK_BfF|V8O<Hl~tl4ATULx`K~UCTnI)wV9O&@&fMy6%KEPubNN(a@@#O{mf0gc@#3ZcdHOTCkM%0<G3uG8}t*4BPvXQj;yKKqlv~RZL_WHsI2Sx!=5qog=M};}V(Y^Bo*I8(-y1HpwEgWmZK?CB3qUAd-^icGyJr6~3HO_$tm3o_ghsq<g3&Bo$`f5_i)osQb4hSvr%ZR@Ub5F8PCr^Mvm&+8PkpildX9O;IYI4H`r(;LsC)noRvTe#|MmBIKuk<RQ1$86Cg#Wdb|l$rFtlfdi3`1ip{ZWn(vmNKHhrWqtH69YK|u3Bh`hV=lxnVx8&&ri6?#QUCVul_ZT6Hv`!~+smpvbXL~raFfMBNiVMScGU4`c5%sJMuhe-d!<63f&Q|Vj|Ng9%1}(~cqQT8QN_FLDI*;Pl5Ub`TCiO<btLRk@Yc-qEH%ARZz(B$!rHB+#^N?xm(8-Nj3x`qqvKnE!%65YV&HM^IYRsV%NF`m-Xx!HmLYPd{kX5NdEcP(et^nx|C~iNLEakk_LHp4LNLbhcXVT2A5uHgW1^`r*`{SM&$7YvoV+3G9`LT@B1ytKbu*!ww^AqL3SnniKVCCQWFl8lE$4;4y0pUS>8Ujt$u;?Ay2;7&@GpM<#=n2#-*f&Air3=uEiT{U@+~gk;_@vn-{SHuCkCpW?)h8xW!3WQ%v=)sU7u|5Xud%l!SZ01R@Wm=!LZ}OwZF`Np0SXumMalX7rDkNQmo*Oa^p2-Lj&I#UNav=>7<|WIW-3Wj05+by+lmWUDxt`+W}rVJyT@zOvAhx(Od>JhXKn63}_xmjrCm_ml_ncZzoH~r?(%hZ$LU!?Ox-RaW&iRdmU@bBw$;a1fZ&=a{STox8zuIGI`A`I`~SH#;)(Ii+tgGw8!e?hf#668|?h|LoitKGM|`<;FleK6<V)u<yFOc^_#q^TCe7rW^3kaOE<e=y;99?TCY^ITh^;w+$2%%6O%T5qxy)X>s-}|{{GDWX%u`FXx6gUDvmc&MH!9rV{CHvRvMF`&#OkW=$#il^E;K|-J>B3%s*N=?3*0BgZvr-N3iHkz=(mGnLB}?ZDgWK*3UZ2$|bhru}=}3;;JA<%(V-1m$GT-CsH&C&L$5c7@;C@V$FU(>en@k=yMY!v6L&dq7>4BU)hX7F^V<KN#zT|89CVMQ#YrWFgC65vke=M-QKjE<1$Uy1Cl6{E#aOg95fqzq?YF&n>afvS#nm6-{I&@#G0xUadNw9l^ZC}nQWeMACmV&o-P|5si~Sxtl4~}7soeR95=bxZddSe_YlcFZj>c#OTbq0H4ut+)?8QLZP^<C-L9(r4Y-h7QGiTq#F1a_mrw1Bvn~=+>h7A~ylz5!ES*Gtu4kq^b3H9ZYSfzFMPO(SVD1D@@<J;lTe@}M3Z4bSXTKFx3s%v7J6{rLIEC^?<yANj@;Q$K!EnNrC4ZBWb2piaMoc5zG2Vd}M<zS<sC0XGX+!<;%T;o5=h%ihgE+-3MY13oK`LdS#HK$}kdmpRaN`%b0$3`AcyFq7Z-oRY+#4tS697FK?x!4y@7!T!Kiz#6a=0+-WaOZo;Lda6utzudPb8T|KMv+ub`ozgr|>3o0&g;>@8-{ww~}i~A3SrH^gX(cL|h3szs6S#g6Dn3l|SGi`gU0+CC1e5lel6vEAHhSY!P#MSuv-^ta$pM%4_Dz(<jz^s+^~fuK83w&pX4e9>w~jBp1n%`<0^0DEJB>-A*so50zTJQf9Fy|H$ja(`7TgkL;495$ou%|0DXd!;hgnhUeM)9o%GWC_!O3lIRW75sd?R5D^Du(VL2jSZ*uk2>zt*CbdOhlo<cTVCKEfXx8dS%HC7OC#Qd;ljb0b$IW)+jjr2&<J(5KrFSCl>2x${Up^**DjLQlJ(Mn=*<bfgqv7NBtkZn_{MQ$I&quUQiz%1++fYYj;<Z7;4K`e7GkRZbxBAF-HoHlL3@Ti&FAbBCy}(d`O1PP1JGd2-f0UmRTl(B`A9Xp&>*w4}sG%;J9GY5cnou`6jkEvDFQ?u!4?dVqUrIfrs(lRKv++72^Y7R_wXB|`-pN63Ulq&IgTEe}Hcn0R?;8iFMirA<DVohQbo5uut4_aEq(ifNYy{fC=8Zf3<WfpqiK&>TmW_c59I4~OE3?n{FOCSD5tU&!>?40CLjxTrYSa${^v8i*?AJOG<7miX&aE+whB}i08wXi+C$pg>{J4qQ1h@jc^*RA*JeoZ7=5!X8e>HJUocG=${i4^2(-D;10@HcA3ZV&(#mE!tkCNZ;W=x(y_|?JSt<y1UB!7Kr4%<B6PH7m-37wQW&1n-x!_bM0OL1PQFkR}xE+!W`RIK6@Q@!rzv{%janpdip?R7t=*NvQB`?IOrAY*DOyqRuuIGPORNm(}l4^`O8o6}y`YTdC`q8i)jF6R5Iia*KiV78)0SkCsavV557g7c7<aOMe>KJ!AP!{mO4tJ<k#`(0T+warW%#hJtO#fzCV#fya;1VX!rvJk#$;^KPTgK4YD%FJ<CyunO@p5989aY5;CUcdF81Q{fM`tJDotH$$JFQ5P7PQBo*=_}u16scQLJSqD(>6_LI3sCTQUV*ynv>VkZR3QQ!n&E5^_0dfV6T=)iO+jADg*X3ZCcsdVgV!GqGC|jJ#%v<@O8D3&Jt=z5{HdP|Ob;KcAAjYJ$M@fHOMKBi4QKs6eN39tFKk*V*~X29z?fyU`~$j3jn6oQn+mXG3Fu8;!FdLqOy}9t_(uC;E(C1K+yIBk=GSnN7teD22EX&o|BsK}?Y+oA4##7qE(sRgk+Nr1VC>KFv4=3W!)fngHkuJ5M9^#I%*Ua{_5u4I(NEqRNuI4Mthrt-Z06Hzsc}Ctuv^YSN^SNl>9uyW4F*3UYpp5}^Qx<bv`#^)C41&pf7V%nBnPdzJ_I-#G|!~44fGLBuA_Yt3|z;Yhh(Ykfs9*qqwCyCP&S^3)wB94F#>my1;>FvJ*qVETY-Mkn4;ohiab=2;nA4IJbw4#BuJ3PT}VQko(oct3Hiw&60ES;<8_b{EAf=Bu+J41dwF~@4>MauJk2x=(f>EXiuxudK6x9cU4>E-cRmKyUn@3Wesc8g^n}wzJ#tX&hd?cJwxo;5)Z*}z6YS+o$$f2RlaK30!mFgfir!|AV5cPst$D{18dimbkc}Fdz@~$iiSxC`r#N-Cnnb8QCvvNjC1N&tN{L`eXjfEA^tvR@y_`%&gv`7MN<6$cAb$TOn(paPej=io@QTWGyI{(if^V9O7h=J7!rs<z*g*G$hE)($8xM>Pk==24&yLyD`}_Kpn66P-M|X|1)uviabYZ1Aqn=JA1hUFM*qqkucw4VuMAPT9$s`(1Ar4C^-;GgDqSusN?>Mdm;lG<EiApeofGou_J?xFZ97)8{j^~2{Fw;bID<0sMjsOyI8>2DQ?ts5hmwm34L9y;IYTjs<F>$y?>{GFOR@tDFV4O`VI2%g#Z_YQRB`yarj#%&`LBHF(K=z0(;Q6yo2}h45rM;xro!+488?a`R&cG_jDp0GWh`6Umb{S>-2vxv7fEI^mV0~BOCcT~Rqz7q0fi_61S+J*|RUB9|r|5f2mX+;0u_aljjhkk_G+teP4-pO{@smjg4Y1;@T~q5N(T_d#KC}!&!Mr?^kj4A258njVpnqYT{g(^>C#qlY^3TQC2_t#m%K9{qgC)8Ca)*{Tf~QZP8hc*V4aa7U2av^|JtI;eeen2lhjGs)SQGHK;m;2qJb3>8edFZ#Impk?KmYt-l``wWKVNt7;CE9U>^xYdMcH|P#dz@GFJAGx|MScmQghEogK@9VaY5C~xIoD1))-{;CP;~ekPy?^xF40gQ&>{B#Ra*oYEwgumDsJBdXT_U%6<X*-D&TJ4;4Mgy1+Z3^q}$b<LL<_A(|U?<27g)zVsLGz9mc<)OpF**4ETWVZ_&#_!RcIlplm>$k-KL&&L~#Ix`wLHSs_t_(JQT4XEh9*?axA@#5vd>$fjYa5nUJnk5Dxz^z^e57gOy@xQjX!?ofIoFRJL4cFMat2fkz>jqeoF7+>BkDRIn)3Pq1+Ts;TT9P{v5GWiHy92a-)?p=m#~K8hL{puT1hEID^4I>yT5$frvS4GdEdj%nhz#oR+3n`I+3NLs(_23P-~ZHmQg;r(U%V4gv?6C$q)Oi5`(NX_&^BWPO%yTYKnNwLZvK#$j8~TG?aA(%?DO&x2fOyv(yQSPAFG{S7PmyQ<a<Y_Sn-py&RJMLPkI<MuSSysqBD({-MDihP0c2i1iQEPncMjCncMiX#<0;`!|3<}<j*Gpg==d`^=T8-d#$lnx*GL{ey22v#{FhH^4E6P3MiSpD85L)`5$;e6Qg$^a^OqTOU1FH*9OhraLx3|z<8z#;0grRMh#(gd}rZsDY4sK12zLmnAe)&X5*$3ZY~<*W^WR9K5aKX(-giCQ~Mb!@ny}$t?lu+5szk*HsJw95>f?djZF!fiu+aPPl$KV))axjlr8MKcJ9{T#Gnmswaqs(RI~W5O0~Lc#fxYN{I-jc?kacwhK#$wAL2nnPlIp*)}uG<j#6{V`R|c`c8;BH4sZSCAn%-zKw#mln-uxXIPC370yEDH%!FUo&VwH?j)Qv)<Y4!6b@>>)N2rxrcA^ulQ5@|au#cws6fA+cl5_chzTUq$l1|D}N09)x+V@jVt&bcx+r*7u2(gOW&3=TOSA%JTV*>b(Vp<nNvQCMM!WhN9mrHw)-#}ZM%Bb%K?jaLGqY^Jj0tETxMR~haDlfKPs=X6mL-xru>3o4=5sFB)+=-Y5^FR!>!;ZyWY#EI#L2ghw#odyv%PQUUqPZ^!3>vZRz@`w8P<<Z}b3BYKknlOrJRV|D0^9^C<f(Roch=q-tcoW0>;&6s&{Fuxvc2T+3cVq%nz-<HKGoP6asfUC+fW>hbyyN4nq;g7TTC-;YEak;y#^#JVG$&YSj`$KKF-F;w@PXj_)dKg-@*X|BIt^!>&W*v3y2A_U4W7>hi(Xy(`d1E)(C2ra1&XK#+i~Z#GvsN=vx7Ov9F<T)h~L;3>o78)c{4$1MeZ?htkGiVo$y5T?bqtIUE$f#_{aJf=7*UcV=j^<>GYY5eGYsI&d;AF;vnPu`;wzLZ>f_RjmZhsiU&AvyJ!fIJl#|xn{!=G@~)>wOW17IM;!175mYf?SD^4j5eBKklD)E^DE5o;p_y)zwd(1&QLt4iqAY>wOR0vgVF<TOTEFVIXf<Rzns_Eb4-$xzO1|ng=yso?+E=~@Q`FdYKb9kNso#Df)C*FlzPKXv>-gxCxHT6!xPYb8a9B?X#3jt+ul!zs)bM&$k`kqO*v?BeAygFK1XC6N9-m~@9ov3wy2W%ZkDi3LTEH;U?>2O*q@IkodORHnFSb-lI47N%(KQ<Iz%-9SS+y({txQ?A05!gvN49Chi=EgB(3~2gmw_r)#nYxEL$o~{nMu&pyA9<;9Ie0YL2L?CkEWgvswZe0t*5cXUDYczdWv-i&Kw^u`f01-qC1yp=2<Qc87cho=<?`9jgQud@ISl$J~6s3E^ut_0A@(^RxN+S^NC#;{0rS4kKA#>}gC)YL4M`IU8OltHXPB%uv4+!lfYK;}{yAa8q7DH`!|m`W;d$6-FxZznc6n0IxlvZzQGCI?g$=3i1vYysM;<5{+vBH5)BWl`B$H8wGEZ5BpY7nvN8HXUWnuD$QN!7B6+9ysa9TaVZr0Lj{fXB5vb4)7x>hYc*N$gyY?nV<HcM0iR>)!$<Swz{RDF<ClkxSO4=X&@uxW&H1cUD#?%$PGvSecS@-l+;L=%1=6!~W>;e_3TCGA=n=a_;<3n26?t+OmT+Pkz;oy<j=Bx@!b<pYHx2Yeyb(-L!F#xQ&d~%}pQb@o;FJLN^4!2<ECfD7(^sG_KyRF2n_OV!W|x3!K9E|~T<{dwtZ}NJ1P}!Zsp6%*i1F$bL}oia<NCPCYGz1^mFq*%6+r{mnaBA)KdYUKoGI<d`QJ9hqztb&rvZV|m%Eq#&SotlSXS!R;QZ{G8X+2cT~sNviK#){$n!Jm{g-_;8fu^R_pEJ(fvd!G_Wq|bYAmXd8Uqg^+4bS9WoVSX)Iy+dChyzo|J4c`bEim}?CFelA>8EstnlUl#%qDSbrs&dkiP=C1kT<ht>(a<+efvswW*j)h7X=!b{rUijHQwuM@_hxym6%3sg<8>I>%B8>dmQtM!*&Wne_&{m0&fXVK?u9G$IY5Y(4A9C1Qu{GHtqSv}Df_6&pU+)K?8|plFCB$${QLY^=<P7RhP#uXA^M^+vDe`SscEUoYFBoK@!1_gvyIxCb`mkv4{$_dACXIO<(Tc1f=_v+iKD*$NC*t`+U?4P&IDXa9fNo@+0jMq|&{tfW}A=*t_`5hHp(Lm&f{GZ<^XSz}+vO(^uXKj8`!CZ#O1FkgB><ZibN3&@N0ZTt@Z861N?EqVBuTK>fIi0<GUw)j+q%sp&>1?%uhH@%X3sD20c@F~SS_!xJYct1Z>e<9tdXN1!J4zA);;lC$OA^Dcaw+2mR_zD;%Fn;lmPl^Knk^zF~UIDrHJkjb<IfB}{cV<&G;)*$pB@q^)VJ6tD_lbl{dhw*fl&j=@NWMG}gYIE0Cu6hrsZ^qW);Fs<ni?A3(W$tn`jm0b^Jt8KYLDpmI_++$L=tyh6U_^RW~`}U$S-EnA)Lz2oz;ZVCq`b*G*uyn5w0G(dDA+5dIZy#t3n<g(HVs`EvC}aFEe3RAjT3VQetwEyrq`#<avcehg%Xb5(F^E;ai383|EFHjo}qaC@5O5@Ipa;&>l(CaQGW02*Qykg8OI+aJ<p%oGP`q%@I#pw~<Duxw$Gc3wacxfUy-g9%9nCS$$u3gK&DXd+0E*x=ZM}$V5!l6^jzwIT_UZB;ibO7l@n-00HSNe^Q3YsW1-QaObtO7c~A!wJ+VF3oTN$TghZrutwDDcKD<cd;=?=KY!>yXakf6gmsYMD{!92%&~aCWcd4Jt%bhm6%W2<t%dd^E1oY77)m3gYYVEWmF%22b_dHB{epM*Qk$jX04`-tS_yYjy@NZ!eMm!Tl}T7zSh`+1^-#lWn-Dtt`U)<Ir>9~%M#=*OI9ReZvWXZsm8;a1e4Q#gC$e_Jj-*AH1Fj|P-aTlR5P(FQGblz~rqMv{+e$|1Yn=Bda|3D-M72`<^<eMlXn*hdFS^)+bOHU|qCcadpH+o7hqva|*qF~Ca;{krLyrQ5MU`EQAU<Py$oDllmcr0<Pl+u!Tbr*Xf{_|3)YMBBp<~@{I4e@{kpA6$sw9tZUcWGF7yO_bdcJTVD4C6UHOG?(nRb13pPnr0aRlmN;sp4_g*?`0n~s*;xbvlaL0!pDqE9ng@!oF{$7!9849p32lC@P4B{@u00g(kpZKUxvh7w}cD%bH;p4(7+sWZ|xx<iggIqtQuOWtz?XXtRs*d2z*k<K}VL_OfSt?POCjHxNnCWckoMqvd-Q8yw=F@#S2=-u<ZqsGbU-tlQcJ-qz*{^jwhK<DdhGs1m_=S(UYS6$G&)`>k4wIP-ASZNE=PG5Em^Y_YWtYWv7AifTQZlWbyi~bra^H;B5ym<MR*X>pEY!@?)k4G`d0;9>G0r+Je88rl7X$wX4b{&nUF#HlVxPQ8L@=IyZseAheh)||Lg3a^sCdloKCfvA)v;A;4BtEk7S~BgGWobyF@Hc~cpcn~Sk-a)EweWsuRjs+$!LqFzpS*2mF_~qIg_9ygkhD`2W)g;xNcBq%?AMcJM+3~JiQ0aWXyTi~?!U&yZyPa#{>ZXiqzmSh;aQ|r`;f9Ht+RT8O+}!pxj4hGf>$Sk@|sO+mzM}2o|uaOqx{FL(gS*x9ngG_r~>_ezE6Kq;SAYq%>Rwp6eZ#g&Q5;>N9W<xR^RWHnq$Oc^=%crdv-XhDz8D1+zZC3ib{#;fLGjIQhC0=y4Q+TZ{O~8|Bg;8ZlAjDU%aDVPwPo$7DkjyGqa3EW61X;V3}A56Sz|DP0s!>bDI)qA<9{smI!n<%V(XT?FPox9R~7fnTp_o-m8?#j~>z3l6EvUaGIjF@g^lb$`us-XU_nC{vm%P4@$n!$>bEbsSgrVHz$EmEfLQ3V>*<MLagAalPXVEO}_AyntX7-q8e`-ZDPT*kR(w(+CYPIb~N!q^f$zPTqn++4#2^CnohdGZgW}`{g9eO!Fq+!r7J*0P1msA<yG-H!;0ok1adhvPk%ILf2InNBk^kjqZ2Wc;&Mh*-?C{UzJ?Y*;a}%_LBV2YV?1UM{8@ig*C+qnVR6%@Ay&itZo~Ud!<DaWm!Qo3?)gh-o`QZT)=K4EaKGR9&#ZfaCN3Ts9pV=N;<w(|7Ze(tJM+9pHO9EX`O3a6Rq<DKes?|HyJbnU)*lg20KKo#M$k|CH}Tn_=LQF7QO`cmkyRP?r&K22c5OU?t;Ljpm#dWqv;MS)x9up7M^Jyws4s5pi<UVO_MdTu<+_*1J=(Dk$aR1yRjiCssvu)IYOZA=rEb=eE0u&CB+R}d3-)RnJy;GkTScZ;QqJ?>JG!rAde6|Sdvy+@BSOj$6&okS`@TG3zADEiDLYU5(B_Q9{%nzevo(?<*Vs5k$xxC9jm=lF?K~tgF_CJ65&Ht3iG-CBo~<SW9CrIf!o7$*c5ex+Zrd!0`<>Gh8#Outpt7^oLZE<bj&PnYm^%mn(=||_g-Ul~fjcqONQNxN&r-xXjF9UuC!--d967h5XgJYKqz<@!B~@%gVPNDp@Xum*`~20fZ+~f=zWe3n+Y^al7ni&v76mzaF`HbqZjF=(%VQNl02UR>v@+1K@L3~8r#f*Kd`u$JV+3_<s)@pon3;<@M`@Oj*xKb#JR2oz+Io(MA-CYiaaiD_LgF_XdtiS=?vVRjlAcARRpXqpxg2iVnw<{u+gd>LQl5x1K$7XYV~meF_0bJBQJi?W(}fb>0!f~h^8AfeD>P*=CUx!Ums{1Oqp8rR#9Pl46NdADOLP!i9H@ByDomj4DjRD`V8Eu=CRu9eW=>!**)dyag!>{#9Z?pAy{Q88Ry3VQ#D%EEi(L~a1$k%ot&Zg|D!!tt`Cyl5+6^ghOXW$ZWM)ZF$6nY@!%FPDXVbpr$>}(6N-u8Hn$d9DKL1hP7m3b;53W&{_;3;|$ZlCGp4pJ)k2j;2P|=WV>m5L1ROW@sv=o2<#fo=1f=@mnj97AYLiZ!&LL-6sXmX9C4aE${rs0(LDTQ21=L+Qm3N)&WqyQ5f*_@H4jXjD|A@0+H#YUSWK4zHN*;<<&&(;LDFU;#p<Odlxc-~MiWUL9`0lskUQj{KpT0dLkaPAF01t3nl8s8F2bS|2zq!drbDFt>(+7!Qa8IyMErs1R{qU`X9(B!y{J7h#B?2zD`#$#MO#c6pNCg3WFND7D(1cc@h69!$0@g)ZWv}|&DxahgvCje1{na<rgDB5n%Lmix)nTRi}l;ZrAzW=76QPCYoWZ_)bLgB2I7`&v<#E&@sz$}?f(Rr6R<2=8mliGNbP({Rp-4)s>MAfd!^tHQ?W;sHv!yLFRSEOmh70?h@GXqXb_APSmh$`tJ0Ra4aGZJ-SoQRis4CS*mA&b39rh}Jvt&`0cxyuwsE2?GhIGozx8V2QrIV}?dR&X$ZwNisnWiF>80~qKz<?Jz(5;s20P!|m6q{J9{vp5@+>jNZ`BFHztr*rcYX(!b0T$5-EITRcmV4heSI(HZ?YV-$c_LowIPOo_}Bm~}T$JE(NV0MWg#>xUX9g8ww)C4bH?;Qe>17G;+ih!MV_-mUr?$k`DH;LNQ{%y&7%`6=qQ)+DtYk4~AqdsTY0^B2+g_ghKS!>XX;XU_rECh>M0?8_<^iOC>(-$$nPe>2*s<Bw-gBwOt*H#4k8%#@IW&}TDGEtFHpT0~${;ARf{oh}_y-tU1lHnb}P63X!KXG%-F|sE!?{gH<v5JP*145ZTaOH%rg(~IBCNTzo@%m8|mcz#+%?^d;5=*(Uxe-;js-0?WV{4<m(cS8Fw>K-T@<w;N6>V*XPs&@{t#Y^E?Tz8^Sge%FC2Hza(M+Tlw?28GeM36M&DQ89!rc2L;_j$~Bj5usbPCu@Ni(UuNqA8^pK7!rge@<jz@}dI8E44#F5FLt?S9F<?Bn}J+y(Mgpkxym4My=}&U{szPH>e`#0n69_9Q6T%g@^yYyLqq0D`A(h~$WT&?F?qI(?fRV0`zOk)8m+Y)s#&@ognSuuLXzz1ns`#@X!8o43eU1y;V_DIQf=nkdqmjFB&kw)GYuVW?$i(wq|?+PjV(2q~R5<LjSemD&n17bL|4kC4`~KZ}V+Zr<D$r%mP3%BG&<(RkJ;#%6j8P0JV;0Fx2UIxX&qn<ijBeE+M4>DDZ!ZG$nnxRf%y$tQ&Oj$xQWJ2;}vR#&^tsk1)Jbp1su^CSzIEG+f#Y`6N5_6&aeIynVW#9*?q$*rcCyMcroC^5is3o#(sXh#%c<Z77U<sd>d37rO(uve3V5Fp|}=|SVoyO)iVmq(7jExhySM|0Mn0!b%V+FwPZ^`-hU`fRuOI}_*3P%2Bta$maYG?=BS4>Dzz5>R{1`G~nIF6@_ea|2_tCR?ldDGkUr)Tb#m2*0z&SD>R8<S9gWqG<S@OWjF(Hz7Joj+#Sr2$|1!a1_`Jj<m=Y*WE?ZS*E32TLh;uwG-^4)akXSCD4ZJ=r)EwRS?))z!{)o?7;;x1tdBXg-|prOngp<Ibm3A7jy=|(PURwbFSrnbLtI!z8bo_+AP)GowbPVZw|&jF;}}`R1Ec9jUKxCvF?>i<toX!zafi3aaQV&E{F>+aV9RcVx}~Q<>I@~-e6Z>N5Wk<S&J4?A>cUQpTPVGvL5bKAx?Dru!kH%vrmKSN5hNh<t|M_eYa#RyK+&il)0Bub8<0gF5qVNkQe)I;Y)2dLNj>Os`#nVRQ1nNJRA7Mu)sd)m=^3%0&~OWHfX_yGODtL@!nuOB5DG%DWX=HGi42;(qKgAg?OH}e=q4rEF2O(m*agU1`b`KW0z*Zfx7+88ZPK6<-0W?a-Geo*rm{+1|Z*t9paH@5SmFI@eO-m@Xi}XIbxz;K+skJ{p4J>2}H4xSxlB-n_*iQNRbyCA)%DQHPR;aPLSjC1M&%0()PKuUBg+Tb``Q|u%@0tBau(4Dde?CpHExFU2)v9*HCUO0YD&2-^*}|8~p&|*0{}I6m)&-W?@P^N}mN9ab+!NcUDJ(%qvw-d$MbZuH*J9ary}(WISB`Yl2UE7_Nym4>#Y(^_w*YnKj|_JOk(wW@~kt^{<f&&Cm-Hb>-{iywni<?Aw+4_|%fbXGFQLIYcxOFoS9=HI4n>g&Q^kJn<n|;+&%5;$U6Vak-0}jae}f_p~fIHBXkb$;gS3YLNTgj|fRcVUQR|@IXnHC7KZpo2`D-Ae>3;y<p;kcb`)EeMRLGiMFFjXUSMDB;LjClFw1Nx;lMdz0<s!`V7-xqJDE6M;(kj^05Ywys%uR)0VB;iTd6}<}Hrr`mXl7o(5;N^OQoHFv1xpOQWLq!v&KGXXaQ)WJvZZ*Wg(cSS1{c$AE+J!h^_pt=<Z^(SRz4xQ%K#kW0#I8_x!r2p|Bpq2L!nf;WM%vTI;m*Icto@UHy6vX;8v<1fV4uDKU|c&iz~wL-p;YQVRWH_TTG|9GQ#MJRwf?&6Lo#AEo~H3&)1_ujsE{bKL*rT1p{^P4Z;XH14KuDUWW*46!tPJjM_%%auZ&w6v}rAqe;)_wOg{X<XD7m&Hxphdu~i_r@yi0~pfQUd48cH8_!ED74tr%0Vpkx|ILuJwjpD>zIm;^21JE(Qt?>9i|o)$7zvgKmfiRQiuyCd#~ITnKLw`lnoavPBSWjo%rq7B0`;V8E!fbakoJ%jc(RX^|k?UKIjAtKWL5wsehLHD<b#UtYd^FJ0tR$MNHLcVD$;2mbC@>13kL?q_vY_3Q(J2AWNZDWsD&E!Rt7W-1T<yaD?22I$}N2FT)3_1WQKZ&-t!CpP@M-#%!uv@x3WE|6lop;WnuhbzAnKFN0%&{mvF+cD)I*!a8zgMx{Mt;wxrt&)+G)y}qInEsOg@D)JpSt^~}c9$Yr1eX5{pz{|&3AyA4QKRvt?tNb5q4oRQ%=|xVu1?M'
INCUMBENT_SHA256 = '84a64d5870df90f68ccf566c87f4dface7770f5597c082ce296e66cf455882b5'

SOURCE_SHA = {'STRIDE': '1e2c882d83dc8d369418d071e95c3fc7738555398bc6d1c35f50908b2c20ad7f'}
CUDA_SHA = {'STRIDE': '1e2c882d83dc8d369418d071e95c3fc7738555398bc6d1c35f50908b2c20ad7f'}
CUDA_B85 = {'STRIDE': 'c-ri}dw0`F)-U`&pQ46Y&=PTsWji6ngnknsz{90kBpiBHPR7R<*>U2?j%CLVG|y*$_iyhisY;S_X`rWjnpxAtlImW&cJ2HA=b)#$VQ&zHUGJAyPn!EL-~P+L_0L&EFBM+AuD-bSPbp!ZRdDN{VbmS;Zr!?nU#))S|I}pE>Yld7XzR)7bUYY_tJA@0s7|6$J67KGNIJf!&RfG-7%RLu8pTu9>P%;?;qXF@TX7u5B{d$-Vl+d;ZOw+$SoKB|E<Oq4sjpTat9PwQ>oivP)!u1%iWl)IwV;l|;W(U7XWj5@){3Sjv~qgdDyh?UYcd&xlahK}RY!wp8go07(QF!e{4+ioMA3zs45B^_c0QtEP6pIQyvakkh+3zE&gytH8mhflPyh9nI*+HV>7Y|maTuz0I2_Hnp<ZiBUnXJeRC%3oxfNeT9krrIQpQtB&HMD*^uV<~?&;C1d22G}H_!K!_kM4c%koUdlTmlp2`BeEqbQ!8h7*~?ENYIAT5*Vn(^1&$&El|&hbJp5YS@}Zog->?bkb>3(J~r3uWr)fwz}${1UBtS3ngM|en#MvEcPg(EosjN!>+Yg9obpl9`DWeODqFmP5-IV-#0IxmQ>j49I5$WdZb!Wmw(VotJSA~fOqfT?C(9<3et^6^W0dEcCp>+oHW$VN2wl-sFqJ_eA;|iY4M;MGWH%#Y&xKoC}~(B(2lt~svA>f8bODaB_^;r9i4~Wvf6K*gwg7_MeTM*-B1l;h1F?Aefr8fL`&Di8qVvjV4c@#L<a^yHHfESG*u_zBnpR{3^@Q5+LPg^KMPCh`Lj1~f|b4b;Q8wh@>E@kMw3$p6M||QUu&rF=00s+`PM%NQD-=#Q9SC*x~*or7p#|$9$T-c7vr!wowNqi_;I#0Z50leTOpoy39|W}b<&?c+xhTZRh)Mp_MSEO-)sW|40;KUc)R`NiBk+ORMlV!`^)>+zidBMfl{<+dMP#)j^K1OK4}gJWQK$eVp`#NMljV5393Tg=NKn8p3-A{OBMZ(T0{@K1S8c3S789tka!xl;@Jf7T`8~KA-EW+QICN5!e9&E8Ru^BZ2SG|f9a9ZFbzm=s@?6q{Xk9|kCFyUs0QQLW7mTowy3SjPxLG(L=7S;==4HIb@%1ly*=FxJ#O;AWy^uqaH<Gl@v&f}L+ue1(AZT>sEQUtFeD%3)g2YmDFax^DV$FRQz~)cJKIHFKKbRfJ6(R2B6fF>Z|H<Qbl&~??)@{A)~|R;n%nQ6yvRTXe{g%^)vIMYk`{h>tGu+(V^vXi?<#XG;kVV1{-Pn$o#B&rySq-Wcwx=X9^sQ;_V#yn+%d{K>vz$*@RvQSlzVyz)4OXyZ(e1Kyi)a|ruR`hJ$Z6(c&1sJx%ZD`Ret*|sp4O{MCNht)xO%=QX6>whiv9<^84*CIG~xcu;1KQ>vd-`d+%PqbSKPjXvYA^awo2)H-X^fa6t%>&I$qVW11ATIvo*WC5Rpm2U9v9(~)|##a~wWi#lc!fNzFhI#X3$_lx!r>)@^*7N$~E^zBC%QGHY!MaB5=zjcbrtKuWF^YZx{ULog*>|zKmRAzy)xVhc!{k{F|{g+QvkcT)`uM_}q0sL~o{M0_~xu<(i?^%lD<n9U=bcHyyg~oPFP!ZlHOa!RvjK%|ff70bbsOmhNTo|gzprc~EviGa*yBnTw5)RyA+QnS@=tpQ6Rn?yu%f)DXJPy0=w4U!}ThC35=A{f_w5jUKkV8DZKB3pK0mWB<(e?W3WHhInelXR<rE2k%4L!y6vQyi8`T9fi>C6A)(YV{bYBqb_!TEqbdR~j@5wGpLxaz~l?f388OBQeRl^_dGg%`9QG{Qa+Mk+p<^?JjQFR}4-(Ta&)iHO1o$6S838%_p9w-|zX5Md6uNQk28gC3d07GxBR0uRlXPk;5Mm_U5g8#Y0IdX+^_|8GV!p^uLrtF@A<;8YIdu&};o$`u}6q3N9I%d6t!OZQaf5b+!Abc%P9n_8|>x&lvsk_);yCpvRHify_gA@J$4T~Qb9S<hVKEusWz6+OW`w3|d}`Gt_Y_x3%36Gkv@O{Rm^a20nnHPr)G(GI8ckdPUA1077aIPa4j8YYp`Q|6*4#>TW1w}^`f<8iAKf*x0&Z)IWs+56X7@TXVBSKiYP@7`tZe|e=Ui=g8EZ#EnjHE}6dXkZG3Xi1218jps94zPsx<iq~!Z6bJ{5J3~ZAdJ6v`mo|}s(Co*AJNsI86X4eqljkgh35m>g5F?4TwDX!1g#a8jyH>FAOh>v2}t{4Ysz$%;SFM1H98waUg|c6Eo!2y&~eNplF(bHEzNN715y2i!3g{HdV|hSU;|?2Cef2c_M7?BG@E-Lc6VR?s&1EGS2s61m|GLf;%%qUFYlhc-+ud4-7cv+b@$Wu`{$~pZVzwM5eurj*)8yIHJe?`AZ#{)QAU$aIEaQ+o1V`{gD$OG6X&0~0k7GN&gmC{@ibV!rw9=_Z`8R@5TsNqKBdaEX2h6zx8rHKKOD7N!^|AYZK`o_r*cT6Cf4gtwef$W+iG(Yg9SpneRSKqHwk%H#*=W`B;INIh;D*GA6#L4G@4BBJ@)RN`+hRRdVYp#Q{9%8zwI|}eYy68d$p>^^Wu5(uba-VR=sq7z1!aTjfU>xLK{4mz(WoezWsKqdo8*{&l~FNSJrCEh=FjCPKmn1*_)~tPw9KT@{myGOY^glDMN(UmZrc;>MaqZcymwHs=0#uukmr8aI<Ib^`Gf#rbXA}%g<Dco+^YIfPq&Wn#KFL=d0fm^nx?KaPMzIO4O#$b+yogC?M+4plZTJA?^duZTUB$NbIJl?}YzKND4nYq4zs_>OA)3{@7H+wel!BlTjS6a)SleNrGOfx~Ba>`#^fOzn$=!Bcj3>n-xri$#O4q?<idGvC!%CDMi6X0w#-R+OMW80(INm``6nio}W2!RF=h73dgEqj#YHqd%N$PnUrkUYyQtKFW)Y2sG4i&b*3ff=vgCcG6ADkz)NoN%+?Ukv<29@Pte#fLIolNU#ZT~EINr*N()Dz!-U)bfv7w{Iaeg&(G+v}C@@|dnJ$?N+A^Ot0GzfKhBG2EH0g(!A?0m=P>xoyFbx%~6V&LZSe*}AA|+X5Veb92r_GOPH^hBy5a?~t($yQcL>k$I40SRU#xeyh!z>d1(;uiD(adyH6AQqi+F@u7Fa`>AJ7@NBE8!%U%n{O0p5Smq*Xu+$Kis$$Zz!9Po7_%wo7%4u`XF?3pBANrcQ|D)<XLwatTR2|9*#OEai)<E(g?RDO?;3hxFv3+pxfr_U!GH$hti>b+Q=@c7pO;%k`g}F6A!ec-1zA%o)V3ys%z`{#oO6^y^Th>iCRTa+$<iBrkk{AM(V>5NSALf6q2<i+!Mh#7FUG4+m<+5O<RMZkUpS@6icU>&ymgG8b8HmU+1IA3GYQ`*czNl`^4)h7Snr)C<9PWNA6N$1#_Z7G|{7nhiI?FFQ4OP;kip0n)7qoQPX&XsIcj55@}h;&&unnM^u@9bqS24J>Y=~?xZDaziqGmwygbjvi5616>IO-<?Cs07b1xVMtJge50Aw2+%}jHdjY5y7^6+JfJ}J+YhKta)V_JM{VUC}woxysXY!k0-NRY?AN{dPHwu5eFQ;e)z~#2{Y7el#w%)Mipw#Q8MnemTk^8hw^(DvW*LQR*sSRof%_j{|O-y9W@^DXhpR}fs1N$F;SqE=kFRur|`o?-C*r-<@)HXK4V58T#Me8}45!J1=E}+R|hiBAr{hpd{H{2I-@<J0HeR`9twUe#2ldHAk)~Z#T^m;g(nt~aU3BvWcoGs!Gr%gN)uSkq_pBU*#PoMBDEzum(e=yT@4(JxxeMh@=6i&jlLqqx!r!c`JdyOSuc6tr7+N+~U7ncUpDMqH^)z;Ok9p1*1Ro+H_c`vi2z05S$ATV5|BGW;CHk!o+o!AKSlAfZaJw?(})S{zGRLcp&k?BWtS5Tr{(ssPG?YLoN<hsR8bzn{#rgnJ1D|C2}P+o@z4{2tH)LCY`7X)V6S38YV1V5Jhr^G+x_$=rM3gbz8E;HF7+<5^(mFYOzI6OF^|M>sZE0w2)L2+C()bju3dZigL9$sv-0Dl@rrm^lQ3b91jYs`~5=sQB-$2@PR1ZqOF#@s%U7<8g})YvC;eAK6KGLkiF4d<<kSXGoa8n@1f)}^c2$To-?n55gAUF1fJ;qN6HpUmMEIZZ>RtlpH=a8yzefn>w1LT5T@So5XDCe)9&gw_N6-GRg~k?Yh{xEM^GC2XXXfocmtg!fvedjJd!RtClNAR&232t-}rg<T*{cF7bt1lV5CKxt=LV-rrakeleK*iq<k!C;db2?*)5G+RQ~ei%z5eX4a=1$K8qBhy`eP6Vr1!{lUv(&B$;RoZ>xTvLqA>haI^_7n2NaXV}<|M3sQcIXihymSwUvpGC?`|<FgRy{mmlJijGRGKQ2D0$9?C$GtNyRG0*|37I*PmcK(O_u<dKpu@dmM9$R07ObO66%BwDv*T;1O2RU>f@fC-Fls7H;+RCqZ0j{^vn;u(MW*~6zlhK*VFKZVGEk__62J$U6EcQ0*vJh(F<YRRBsZ78mg)FgZg3P@?*@pK)P#)q#{yG6LHLh$!|^eO(2fS$#Z3IlKIhLh!3juLkr1O%H>+Z7x!kvVTrEs@`L16csib5l+;GK9IO%glM=ytxm>N$BY!os>4S(&CzNV3Y4j<Y!8#4YDac1n)MNanjn0^lB3-b7JxYVEIxr#tg*#XnM48syv8%c@cP^=UOo+CqCJP#A8@pW^E3Owho2-BZe?m~T0E@8LQ63{|U5;?zhH2lyoAc>tnpqe#Df-eH@&L0Hb{Yj0cI*n~$hcb^RQ2M~?$%w?6ivC(4X{RA8t~O@z6g!$7Xy)Ou?w)y7T*X!rM);jE2$p;OIst#e(UFsLI1r@GzB_o*cyJ;G#WiTaEs^_iQSn_E@*XT5blarTsKI=!``hv)+$KiKF8Q3$h+uTq669-)1{(2MW5?+LIR^+4?h|Sq#5jH-W)%Z`=g|e_mW>v%&#8Pbn;|sM8~^yn;q|&XTszS=h$P|{mCGaw9wG0OgL%5WCe_uz6|I(r+IiQyaM{APmnSI+rir+@4&^$tzq95<A&X1abXOPIZeTX2@R=ea7-9k(8Y++TctZIqWiU>+1B6G`Q}m{l@5{y8DDosd|^kgmRuY1dDCW34ZuP0)(MGE>z2%^qkEHXj=@w8Bj5AIj4=AMh)Wd?x^iV=Q&?UNP|~loVu@Z-aX=W;vV3vh+gQQ8eOv?rS4n;1BAogHngoC2)?k9K18iqWtp<wD8g48ia9TsYg250BN27_?B4)5dsD{{9qZ4gKOqoDHB3eb|;Q(luVA~{C!uBSkQ$n+y(GUwQmpb5w#olzgfnnj5kyg-WJZp<V4ig!K0RK0j?e`&Oz%U-E>c6YRt4^M)|0ODmMpZ%s5rz~lT0}Ff&}0b7uFl3+2>?!KL+ZELX8}M<`2UU;0~ks3<hgG#jQBJrvXsYS&dwRVpypA$O3%&7Cr<>{{+NnU16;7Dyfb=yu4wUnwYqwSw?Ll!iM|o?_qhP!Kq|4a!X>o0;p_@sM1r;3G^whWJkqSvPcjpS8mp=akN$|O<@G0SVdzhwi?~j?w(ncxFVM^wewfNmSYH8>p00GST?VO*e|^T{`sL98FOLSk2#ngW*F;N%-(Xn9k~w^M@#e$psu~eIqOw<hQE9%Y5TKM_1kD$LfBjV)e(z;$Hix5mHF@n_b6Ib?uGgOM(f6P55fOFYf5unpmWXc_ot3_cI@WYs75gEuAFB34&0eBP>6W_g|I%7Pfr3!CbK4l(mRm!<Bg&`2dbDvulrNEcWx}ZCUaeYYt4scep48_%mC~J{bf;RnQ~Oe)$9n0`TItSu>COXw+$i07$Um%NP-EymH`VQx-fgdQP&tI6QGN&3@2d5?cIc<oCbzv-1>f)^Fh8p1M@|1|n=&->YiU<I<0ji5(@qJE1ieB@BV&2f=QnL!&F8dKUEk#wV*=9zto>%Qb3Sb~nUWcFn$uA;rkgkHdT7@-2tED;#UdJ0KNG9L65X60*O*3^JN>dWqV>93iF<G-I4rRQKUL+@%ww&>#jy2a#GkIc7{t4r4yGPt)XPT0C7R6Y=x%+8qo>3v!7M>NVhz*d>-08nUEUYIOHUbfnD1Yq;LY>?st+mt#M9fC|31xc!RhF{JX^0*=d{SO_}ob2Hehm#R(a~6jwx%!wI=N8&U5d?C;E)$hZ{IGHNJaSt0-S~7v%*+^=$t+E_d5~U;Z4hY*N18BZg~5QCo)uHD(qxf1+-nlkzJ5t#CmID`FES017V%AM0k*3)X#C$jC6>oV4Z#L<nB8h6xef?98~wr9?FR3D~tlwR#X8vQ%Xjhm-h99m%lUY<13NgGuOFBD1mV5(=E>w`JHZ>pJEBcxJ2}VBUa@!nqJwp)H!grUAcbug1<^oiiS%89ig7ZuebX5&U+-Qy8(gx~(zjU?wkLhoO2GMkl8&B4meQ7*vA$bo)nQ=`PmwowfC~u=b$Vt<~2ataa9U54ydL^=iAa*4t=@57vW+l?NN`3W34)7^LZHwOsMpqJEzRijzqgk4MBfvc-Hjrd<^0kT#f!aB@1E3S%^YZC-nD915KaE!J=d1qv06%U2${?pZy(gU`|n=B}DloTtEfsya_Ki(+LZNz&pQg3!uhBY0B5e=}b03cWj1chwty3WS!OMOeC{L2ud|dNdCDe^Bw0Qq|?9YH6t&P_Ku!D~n1c*uj4@khn1K+NjTTN!sRb<iF#3G{DvhCZNQn5l!DExET;@qpheMz~`k8Ta)2Mli73)>Y^^9gJ0qkb64&8ja)VHz*$-$!#hj|P$*j1r6lJ0Tm0p&s?ol(SaQp<U8Fm$;k5FMK99?9fO*^xY&YO}&j=rBGZwW9uHwG(^t114>lVA(05HJa-lHaHu>b|9gUEYU!nYNAneZ?e=Ofd)ulCCG&H>)Rvhwi2%mS#e(nth2;%%W-mk@@g5>)nk(!LSH5phGoYW0qoe+;Hf=@X;+jL#J@Y~Tm0YwMsqK5Jb%^dD>|>0wzm2%@w;S^EvqKO%|<6{B1r5MNA-;V3j8G2bp*QsUEE;Pa8Xd$>x=*YBXR^t5iYj&<KROa6Ah!8M>9RN?snl|5{vZOBg6sroZ49&O1#$CFn7v_-c8LGX}E!k3A;(-U4$`tt~VHi$bs5!iCu8`L!@tx(Z>$q*F!Kf$I@fk)`|XD}b#zLz7WleX$aPLIn*IYRs>)di|PZJmToeD??=>HquE&6n>w1wfx3ep=72d45pG6hHG^>0(}`Ljq1xo-Z=bpEA#L{rshoHchv2UHE+c0_1@5tjpT>ymKO#)niR0@WUf&2$~|FmYH8{s@1@^o4F`zCa{|cvdshq&3r0qrfN4+%{EglXl7p2OwDelmTjh%YbMW^@$!Kw+bct~MSFKo)en8@ELqYox6HqZxuOF>YU7p#K?8w|2K^u?z@HVE<QZr{k&tgDKz0hd09{}2u5_#ibeNxam`pl!W*nx|{gR51Q^T%?J<c}7dvw+a+WF@r!l?S^$`+%|&lx;`naOv)53*<it$71(2ZKQ;eliCP%&w3Slj9K9a#)X3n2<MH(M5sze5m&j$19z^+)8}Q_hOG*@&&Om#>ZuGcn}_9KF#KMH16m!vOw9DLECbzQS<h8u_QY|lh7;Zi{lbJG8a5D=W7UPWX>i#2Xo#|9=N@8GTy@jd^i;29CSki$Lod$LEeO@GX#W<rk4)@S2*n_?m@aq@%PadR_3!{AzpFzRhW$n)*HQ)R2K&1Jw04nbYfe%`xR}`YQ_ZLz%BW-DkBs4g>FD*v}M1Brog<W-|5PZs^-njm)2cDYKA6F?PXSwE@RlrZL&hgqw#6qs*<r;Bw858L%!&>wA5A$@4X6ji8|WtidIK!NVGi)(F22swO<KQA+BHOI&Uro$180rsI@e<E0_mYoYi=GF${g7J#PEiBNz3K1r#%{W4!hlg2ZT2-}d-z>}JEkkW1WG8@?D2(%c_`I1x+3MxuGtlO%3J91g^XQ>#7`-CBNJXkELm{cJ|VmRWi<{Co4#;cKr({4|RWX@}HGFq~F@0Y6dCB^C3p`1p685ObB|oCH@Kl->Zxq@>QPR}_r7piL%Nr|L5Xk5fhCiRD+dVck3K+U_#~TIy<qA=Mz>isCCu#!@mpfV^!qlvTZs$GoEZymK+X#-ci_cqXUu3<ie1V;sk`bP4y$$IlL^(G_)uGs(9GUhY{hfjsCZUAtoNN}r!tpF!c{Ly3W4H7S@<bGegNIJP($n~W+h8dY2{DmgPes@NHo;Ei-t8YO(OP6W1@dbT>vb!d>F>A~Pi`v$k*Mc++SIy&ApyVKyuJqG%Hsf$Wt#*Hft7o2U0lCn05(T6%}#fSp{#CSFfXQ6*9wU3lHg61}9{F8o2&?u(VqC?hp03H#|h$Ap<wdqUyLR$fJ!$ksP@F+`gR!Vn=pf(vJ+=pQ$o*(o1TwSH#DM%$XdKKB-UWGxA=S{yI+QmINxTH|RqF&MhESo(=3DX?A4toNQL_??t?3q_%vsfBZwt<AAfpiszj{Xiu(}H7nY7wN^*i+CeTK|pJSuxZKTEZBD8SHW~Y9?=HK1=a>X)9c7%c|yQY1agn@^p_aQ-Dz5zluyOV%~rjG6bSSA1+z*MuOC!@IDi|913iIkbZ(LdtiM!(`YJHqME6H&et(>FFL;8mio=No`s7q-$ve*noD{RX4povCF9aiotCQ8l>qdIpcEfnPYx>OGVzmaH`G^NA9{iNu*+jJqX=4>$w9STu9-3`8_JuF?J~Fpbtlsk;0x+XabJ@Vg5Yy8$D}UV_YzhGN?aBpLVzvz8#)|==%-li>;~(q)#=PmXH;`Exi1pq)tKrKrXw8KZNc23oNtNE7=Q%FbO<3CrtytuaFZr3v^9tz4{VR7N7BaxP01^#%_kq8Zqsn6*!H`Zn?|eIoUl0e{oB>>ob3y`5dG56DWvMrbkY(TuSALIcG;BTav87K2+&-Q2GbTDH!MJthO8ajR2ac@0wVWL#TX!1?}F?*ULjk?n9<QvozylVPW7%dHP7ygu2?pqFFC!Lx0dwLb?EEImgR$aYqu>s$O7N->Uy<j3;3JqIIGfvmoB_V{g><_smiz?`&X*VWR=E~Q9Ep&d5$xttzNiF8c!#zxt*8o@>R=rPx3pWb5PR!>q$kMk&K3Bv-)wT<eC;wFq9nqoZJgXa{C^^)&-Q+k<BqrPne21!DV}t;;t#gY8XfsE4v_v^U}m=EdasE!O<a~d$dfyXxs^6)-#!=HKip8uxI1k+Y7fISmGocL7!GFDEIhcAoi!{t@fuZm7a}FEyt^LTo!xx6}ma%!Kbh}^*vdAdNn=q#ocNJE6x@ME;M=u2+n=qZ&=bMHa8Jqh%Fmv%N*D-g|6x#2EGn2#agxq27Qnq3nIK0Jnj5Gc#0~QfYWtt4~)j}%tjf~>bxR^l{seom)4#=QVt?E2$%<#mx15Tq2Kt1ki8UygPVe|2E<ARR0G!E&S`PagWRVJeA0A7d`3;fDge0h978+@emoB@J%Y9i^SI1m0xkjL7XV|ve9tR><7=A0PN10}1M*k8T*u8x2tVQ`-Dfuk)C9idW&d}gIa%wKgu7mhCPf_f8E<yoZ`{1n#_y>F&nr#MVY~Z@F!M4=NPd!*n8E;i0yjkcoDMqMwDxz6TK^#3VO{!Ak4TRO>eFvRPg>3ln@S87R&@*t?G6SLTXeN8MzmY(*#qej(E@@3ORn#hw%1p$#EJ0L4x5I84XfjU$%B}&nQ;aiL`qxHNY_lCeJ~!OV5`N-n~k}%Q986^q(o?gJEZauCZ|0co*3^hqSrVQpD6dYp<&Pt6YN|%iqd!;n4=B^2*o}|FpuD&x%x`FFA3?l&{skW;d*3TM<Lt0reSW%SvTLeFZY{IetG(Ak5}fw`bJ6K?LA3i7Vz`VzxKI|Rbj=HY}~qaE!#e}Cz%ocu+64m!eNM^YuHDPMf@^wqUFG#xPDqwnoz$WE~1TX`hT>R-TwzOW8C12s%PlH>c7|P5OfPoe$*PChuYO291L1zG8+#s-)ukKR30}<+#*#2U60*Kx~0{mk}wCMI;Smyp*U`(cA)56^#-&R1Y#>H*r=nS2Q=ySfWtSjMYe9EtnEy&m0}is9<n<oK_bS$VIC`s%0{_dSv+{A<2Y(l5YIPSW*u}YT@MUxW;%QQVc!IRqQ}X65Qk~_mc1uWJ**o?Ya&t@)tfLa@%S~N^9tdj%A#JWS7z!rx&zu`OHm0b;GyYvus)-SRu*edDi!=UV{Q+?TuK5e!03587!Nr(L*h0k(VM&zS}{J&D{5SZqaDnyBi;*hG<FepYA_DjoEx!j*irlSvu#~~kJalBTm}Z-gN}3y1D<Zkmf<nGWnxXOxxX@#FR2rlME$-M@VIVf4YYvfN}JGjYtlq)M9=uL+=KIjq334^bea$6V!~3CQBxZNZf<s_t#W(4p2Typ4FJt)q>8?_0F>6RWdz8hM3?7b;12ZC(MgTm_?jgi;?m*WG#U<%j?D%$YnUJ-IxmCi#U*yHpZ{;9%7i#od5kcbFk)06w+Z3h+1jL^$&99x3+|~bE#{^+F4?QE{=a{W5h>O*MX$bChObL^#tb*J4wgA3OE=%`Ty0Q^+jE(Hk-X4GLjrN<rL6BUuC$UW*z;8WHfe(;f48H4mVgnP9A&_@_7UkeOS#<-tdgLQMBA$SXo^VbHfUb8syzJ)hm8KD6@?I``~F0f=}eZgME>|t%2Ioic4L{Vbv%KH!s8*2%k!NXjhX&#&v1;<>b<1Z^g`4Qu{KHZn@v(2UU?9pc<W4?)5WqQag9@(jwb^jOHGAAiKHD4XP^y?eWV#7eb5eyG`h`AzI<*w(VH9t<2zOSsvX{TLpfPi-O2wlGhv@0ZH-L3ke`&cPIeK3i4yir4@9#y9DeIV!31{3jMumNd_+TJp?nFgN;9bZduIb9wB~~Nr+$2`sT|86ND<^0$j}>j5okw(y7KNW`~yUtdHdwWVHP3uwNi{t%t~r%9`CA}d-$Pz-x5vPF*pqUv`0@UnAJqU`=g)oz7tX3XN6B0$8TZ`Y%T4(9iL^%FlJ0-JchB2!SYN*x|0~OLC|k%EbP*vIFLGrT9*WFxZw=P#OYv^v&8akrcK|_&5ALK^w#ULzCcPu9^F$@zfc2iaef`){NS-h6~X&;Y~9&{x2$t$&^3&fI(-~O$g*JC38W&XO!mu0wgq+?Ufznhl@+waA2PZuS>eyupP2?bJ-fvL&xb7pXUj|kHP4y}{t~>Sio{##*e}{yUjl-0C@E1ynAq-BBCEBI)YDoBb`VEg2@Vtk9)sxr-+z#UCi?xOPI4|nMGy<H=+LXHW4g6y8k@2V;ta1DVveq1IL`xG!+YGnTA`mrzi&vD1>%YErrhM%JEnJ{T!)fvgSLIWt_|>t6qP^^1Dr$RV>|r&u3E1N2ng0XzXDKzAONG+#w_LtEW;%mDL!EmV;TO~_)a(kyMeQ3nmf;*S?8Skw!*U5=?#4lr0^=;<1>_?jxBM__;M?&juhpi$a<RWGaW7lF*xfvorx0hXv0ApZr-5@N-B}Abgl;py%lrdE^&osKVfn6oy;QfiV!&#`%AN(mtwn(qzxUwxdk^}gyv-PEtbavhRf`8(H?anuVG?I?7juu2a`0mAt*n(v~`ST^tQ^wLsL|OM!<_U3|onu!32#~2*dze??`bFnU*>YCv<-1quCIYNqoX?BTzF^>4pR~=6HlugJS@;FBE-6{8a)~UZX3JjYwG2g<tA4ph+Dq`9BYGG!JTLXr7o0-{pV?w{C=JU{9d^=Q`tF(@1nXJ4TbW6KQGhUJIc&#L#3k+@X5k-n}8=6OAZhmdGOHTF^2J`l_W$TBraS1!Sx2C+MFbPTn_TjG_-XPXVz(=Q{G8fUJ%MaByA`S-U`&IRqRn;J_?ZN!Abgqv4PcvOBu?$Q~UB@Y_tH#kqr_9flTz{xsx%5DQ6|nn0$bMR#;zj20JA03$eo7M>1bxlAFJ97Pfoj^kh%UkBz=bXAB>eVw_%^8FAp+U0M0VblrBTun1N7E`5}s0@RYgcp@q2xB>pn@P?T9fmP3nP3S&#S$XWI=pmhhZ5U{V>#GCK*VHfM`I{$9W@DejCqu0m_$IYqe4E>`3Gnp=#Lx*ckRzBXn~JLB)=lJ0QF%ua<80*5uwer9`*d?m>>|-JQDdJOV#7ij7XAA7sPCnSt|;KiG!GkP&MKu9Mu~_WN#c=qi``5?bQm#h*%X7j_NA$1jpS|DxZXQ<QJ_)Vl%^w6fOXu29{wb!_b_B!Wws`6Q;`XcSW=FfkeXvJ3NC-mTr40b8z=i5*A@#pOd>@hWH5Q$cUx+)yj%xptCqiKDx$CjHqB6#`9t`o}cCSrlnNDC`$>|Yi+ANH@ZTHX_x28Tp=%)WtdK8L<c1+*szAAuWS9SpywpE3Qw2!ufe(nRxfi%KL*_>47+InO)e}zpN#AuV4w09SjVXp2ZMhM++x{Racq<0V^(?v=Cy$Um|pU4f`MZis{sW1*W=&v-l9uVYYKXk?6kgHWv?0jhItYZiw=;mEwIjwVcAkXosa(tHArMmDym@4J?8g5EH))vOgubPmN|W5CWaFae_r5&M|6<WWwd(5CMW&UvFCHMbFBd+i)|0XM56K1#*&gND|l?UJXld9wiVl#Miwi(nnzIV)T|Kfq8S65HYfs`c*w##+$`)XbH`M$U9YfKAC~V8SH0c_t*r3THXCCYdxjb>SwW*{rQ6p`z_;z_beK%SPNM5WZHa~^A(zAp)VL7ph(U1@YV~{&586W$x~i@gl|+b=1Ynv?j$J$mV~;Il`HI>2GS9WqS0SlV!hb_(bB*;)NGe0bZYHsal!vgMR@_hDaCm(DF##w(IVfYKFN@X@T_H)px@g{E%iVIXgx`Vuc5kS#-)9d1O^^B7_s~dW*d>5nJcH^+{?N|QK5fWjMzFN!o61W-6yReZ=a5?eUxxV{ww<)kI3@#(i_r=inXz-R2P$nfLC|_SIg1nH-^F$(EL}ch<rl)9_xp%FpeKLBnrt<Z>i9wUA-a>H8Jv?<X*h=|@r2Ku6^qxC1Cl2n5voe!Rt-m^li9ck3$Ou|Ugm;hdT5{1b6(cpe&ZX04Xw{<X=N*XqV$O9II+$SU1=@Ip}We1>QkT-VbszHS(*f4y46ocbazW{aHG(dqUah*^a;7GHt-u+@Eck1t5-O(aG;SO^2-6FS-j`~OMckQub9ZfGOJbq_w4f+Ej4o10DK;$FrJ^0GPC(z;2V7v_?q3o)Q;6ZwiuCFGZD0QYz8Vw&9j2Ca{@0d6vA^>YWvwSvUj{EbkGYcZp64OG0;{NjX`2yBMyGlibnB}*lKUR?x$Blk#V+4zFI<d>+U%&Y2lj)z7`e78HlFmJ3_rnY6h`i49=I|kQv@}+?qI1XFSiUc|W<!?k2#Js!fw?vV>OrLWBKO@c|K3;eDIB-Xp;U_-NyD(voYbgTxFmOfY(^!H?92NLyb~Qv>iHIP)e}_fM8dgG9sMFn1G|Hnxzun{+v<R1xl5_b!o?A;=R?X2rE~5sy(q9UK--WWx+<8sz2sUHQD>XLq7bUAr<y8VV#5_;aCc31$%EXT0pJ%J;4i@D&X^75CZW^&Ou&L3hNQYmartc+NjxO88t|{@tE#jtrSF83I7^66KGdhzMDM;TbiMf4M8)rD_a&!RFUrFHf%#@>~$R<rgoPYipG4P%N9~V6!kbL^SY3Bk?osU3o$HpZ>7sp5L#_`%m(I?l+dS2<nK<QK;<_s%ikMid_nUPk&=s5v;8gYb2&EZ$7+!?oE6hmydp-1>gC+k^8>kTTodrYhah(HESmKymZLgrg7;)lo4*3`a=kpUhgoOJ-?^<qdD9b^7&iLIyob`5zXiNG|X{|=QzXj@+bT=|FUEf?Sl=jp=tZhA!?=%7xu#^`(b{C`o@%G<(u88ytv4|_>_H-4w-p*zN41t-K@hHi+eo&(&jTGHMMVJEAAQM1tCfhVgpTT4XX%Z4AEM~Q<zFSrF^?s5+V=k69$)`86`oY0_P0bVpuEd-3)y5vYNg`<hM3Yocl9Bc@eDc%>ACBdtD5}+`v^R2FOpeAnSF%QQS^&!*-amDr!rB%s^R(-681T3;y>L|2wCD1AZUy`+(mE3;=(o?aoZ9g|$w%xE1I}z0a*`@)?`ms->ToKoP?XLMtMoA_6HQe&RJC!)>Bk(1zW}mGEVJHYKq<hHU@dxRp7Pd`DMUTE^FL<*>o8g7j6OUscmrRsE`#zN#g05{%OZZmzp~Wdxarb~HHDZiO6W0Ua$-{BeEOE_QuG-^e>p23yi2Ea_pD^jP$AWF;tkClmPyL+uNbinQBE!^fwBlB^csvm+i_NfOgsDkn<HeQQ~f9iIwBKP1pq<dz~tP{hWu=39rZS}(DszGos?F>y}3L1d|}8o3r&ltN{+ZOv4kv}sLB=fw$A8z(I7<rWrm6Ntl##4*o%UTaV~%R>^Pbwq~;)D%MQq~{=JFK&1bSnmRPCr&e2j_X>c0)){*d3vS|2i&Uc+QWl^b{$7_I!(QN3gp?AM~$QW&&(YTKCE`=$J+;;@X&ZHHj0b8K19`GAELN|eRAF3a@}Sf;i`p>aMkjK-NKb_ZE?n>ZPL~&*6&~D+IAh+I^VzkZJS-%o#LgnP7(l|2g{6Ezu&lZ<?7dNvikP8&MUplGS}FN3(`WUVe?-v<ZF<&r_bdXi{*Ok(bZ#@F8&?#8rJ~BD;%@pV!y7Q-}ss~634Dy!FIx*=^D;Vsn666s%+f6ip5)KkEwnVOEq26B9|vUZ+3L@^P$2L$uJ~riLe(!Gby{W!g5<9$c@7@(U)WOk}nyQ(`<@5mQ~qa{;Is<_7jnKacqp^o+S#cq_WWCVfx-<>jII++X_4DPGH5}aj$_NkLcGBx5Dvp5pLvg_F8WT+XRoQtfDJzy$zaNtNUPGl6*EH=uyN5Q%jGT%JPoU$f{~BrLuBkhh{MuKGccH3{r8lQP^jvbHdk)(O$Z6b;5+&k};mJLNHN#Zb@>!eDt{lB8((hHrdhQvqEhruPiAn$~pr#?1NZASqcEl8fa4+Lut>lXyh7_{ID(hJJ=Rk8c(rSXZ89I`=9?l`=4v8>94DTu+@&PW5w@N(ifUcaGbIa#%LnjPu^t0lM4QISv^Lt2}1s%mtU@z&#5ENp?imC)_+QE&Rdqzkt>17B>-({#g{6NujPq$E&Xw^nm1J(U#96bisPieAKKj?+TH)8cK7>hE5E6lGHIxGMN9P`3eO)3&mRiU9}3U^q{0)OiUm31=WQx^4pRN6yr{~HnkYo;KQy90G@^e4jp!AcD=W!IU8nc}{cw_MLh(~b0rRU)zihhfFRvlZ=|Yz?<TulXE@8@(-fZqX=hCp8M|bM`H}P?BJQsfGV1MXfZ>WQ<e?uK?y-){R|Dl6jy83J1P6xYosSbAShYt2{rGw3@T|azVook_X^}J$)g|=$dXjbpr(z}~zRlkQ$^?DlBKVCn{%;-0XCJF@(MB$t01L5J<)<s6H6G71pv`~S(OXGx=XcLIL(D9VN<v5Zcp^|kFTCgOVX;Nkor{hdQmjYG@G?HPJ)yp(TkCJ#|@VKVQoUF#vR$tU@wROZJ^JIz=50x54hsLV{2aJ6ai{qeTpfs{o(eELTtDj6lJRU3Ma@FV}v|bE~J&8lGNP_R1pgBA?%#nYvr=*EYrlV#O!AAJ62hR|WXFAkVlkRHeay=W!^zGiR)}GO?#_tKYLsMK96_52I@@LkTzbEl^B7kX~bJUs|tx#5*9L?HBQTIV~wjUgwZE=)3H2AX>tS-{Q>LQjuC9iw>b<g+tE-{x0M1AOZg{I@1oBD(`83i%!mc}p>)bt;XZ}%sSaQ8i9-08Sg-x4Rql7^IzeZ<FJ2Q)@TQbL<)>|<K6&_vnkUPLVfdEmrB4ObC~)13r`K(GjWr3-z+xiKQf0TZOhh({^b0QtknR(i*)7Axna$uwDp<TWzNca3;hE1w{~tVZ5dl4_(W!I!^!8Rz{|;@)WU9#&fZCh#~GD2cgTQiOgNzP!Z_UllAF=z|)iLUE@V&xvtOb1}!QZg*lVm0qo$wvI;#=s>*fj*91lPoIW_s5UtCH77)mRRHwgf{Y-|I*~+0^$91SMX;@P5hvGncZrrjRNJHwN8~_2Tj#A8RH}7d5fMA8{5`;BMUR}W<I7E^nwg{(t995OSx<3$<fPEJ;bc6In#&a_oS+vMm#kH<6@-@NrAf+M>Usv5ve>IIzT%X?VYk3YMO~{SI+&+}>C&F$bCW^h8XczBC*Q4)uQLKxEH3LG9}9X{cPa+=N4LGVE_%<8V}JiKaKo%KRUe<*m%R1*QSg|7LtD0887TO)rSr0QcUzr~e_-;H9UjyV?QG^PC+~U7$$Q>%@}5KVnZPyxyYQw=u8hJ!!*Mtrm!puA)e`|iy;YohMlUM_u-eecTGjIDdOg}W3D(QROO_#F;D3}={qvm)E?&9R?$o}N)aN^O`n5*?t<!%G=*^vtlI~k~|HU2w?bJ%|wiaX_a9$4kMUZ__&AzA|`blH>Lf2ob;+tPV@~fKss`+;Bbd&Ppnk#V`JXC+c1K=T8zX?Fle_3$&40yQn5T&?i01yeP-y24(7eTg-AlpWeZ6o;O;H>ro&H!iC`q#jjV4Vz<eP%Gb6QFc};=vlj-uhnw_N*7xY#Y^V8`W$Z)jtk?>p$QZ@LQ{XTlm#TG7G(*8GP>qs1cw_@DMOvt^8eJ+ImsTwo%KrQOmYb`{T~V+K+RAb5XB+|8t>{diGR&=3{ZE%B`bLwZ><q{^P7%|EyRq>e)8x**5CgHtK)ed0YQ+-uS$2{AbRazJ#);?K2;@J5|)FqE3}r#OlV6Gx*1zLF>g@wvDxH8*AA%*8aG2{NTqq#yMVF|MQ$<eQ9P-@n=59cdDp^xXX8H%-h$hKhE}F?`&Hy*0XJ_XWLlMwz2-l;fIYM_yPD~z5eIM5BieNVu#O+9`4jor-nK;mdVuCf8dP2C(f{5Jjk~3Alt@+Y#R^02i}2l7rJJ#fYGX9$rGAb+OVHDA5ti8w+IiM&5*(M6BO6lv<BuDvq@{HPQ=IEFHrDqX*(DG#5!!YJ<!>MjTwzr*s5r>s+F~>Gn$0-b~+g>)Ub6inoZR~L|E;F{*#nRw0P@k=TLdCcYe{D-qvsw^<ky7N3S^FpuBnO7pTmxW%&Y2h}kr3&P~XMBu>Lpx4_a8C%E``C!bPQ{wfRESF+gN`W7s^2^6P|HMy%6Vr#Ui4o=#N4$gYrEG^daFrz_N9veof4uA!!^au$MI_lp93hJ(U0Pk6VQ#e+V4@ej9C>R~Y(&dTRTH!5K<|MqMUXLyeWT1|x%d_JZ1d3R|NGLH5S%?K+@^u?1iKnmgQOa;6Whg<CjC7$U?Zbl}OwGe+ipX%dy|6A9FtC`8W;)0=fiVZ&Wpyo->;nR1yFboB#5U;?%C|Qfc1_A*)4po;V9}gwhb=xN%P*UIi1mj#Nn+^bA?Qdri$njCtnD4+AK5W39-U4>tO6`thrRCd&Z_gu4Lm1}>m*!`I_x=FGmevvKcCY<aFY(C2N?C<s2hYfuKeE6#`#&XiFt7(WAqgK?w*!R&*|?nf7r`iu1qd&o$H(N`ba0_di`O4Q{J`#k6v3R@;)NC)%i}45_~nNR+Jad+B)5~<Z~TFo#Ct-BI`5E5<_I?PWw!fSjVtptp599RrLYFX{&^3hJ%yP45T4OoqhG2>aT35>7;cY4vFIAi%PPT7kgB`dI-m^%3}TLMg{-P<gDw&=IFV^{rkn$#{dy=`qZwUc3$gP9OPp0kgF^z^$KUUrWRlk!#3Dr<+I_7Elj1dyNmy3{FVqn8lNx!E^!GR2bYf^kP!smZv=5)M@mR~9YfuMWPgjG5>#g547<Egty-(jmbV^cD-sQi|D4vB7c#9IU)|j|&nr1s-HLC2-^dr(a7zTa0)g8?zQ9IcU}%XCV@aLS^BFy#(eqj2;RJt1@9vyH$-mU8QI`vna&B`;x$T3PZY=FWi0G$VdQB>H|GFOo{SgZiENP&cHc-tqP_r5+9sxTpCZh<vv+O9OKr#|45nR$xHEpO`*ig-Ch+_qaGbAQp{ob%20{cT?`ADReXkUY4Q-Bwi8ID}8nc||&cpcb0c!+x|p7ITrtON1=AW1hj!Ow%VE9G{%l-t@;ZX-*%-7C>>mi7{q3U{(pxP7I<%_<e{kkxCl(8dxkbsM1tdAtSh3$u{gTJBNs*t0Tm(RF%0fblJpZ$D)SCrN~0C?zJ{PFoj9AT$XP9BUdz*LR8C&@Qn|$^Cna=SI{m&FO?e{NL`IAor%4ds91XI9?)Vj+MuSKzF@SIRkRW;tazc0f0)R3*wjraBa3?@t5<s>AS)k$jZ06M7yq3%9WnBs=k6_+*DQDkpdRdOxP^IYl&wDGlDVF#1X9^Tw!NKbTvoQgY0O~X(7?gn>X9>*D`GXId1K1S+(DlA^fWeUQO>hMD>QBO;fUQ>)X-2Hrbo+Ry>`)hRyj-yi9&U{t({AEJ!{EO(8M2_vES9N#n3obRZTcW4%rqASR{ku3JN=CBJ+bEW$6-ZlK!@ibryj8fGMVWinY}_}XTF^^O%>rZ8Ys%s{ea_8g!lK3Y;S$W*ijduc`4BpuAA;zYBl*x_s{*?2a~*I73fIE@8%V?i!BP8!^%7zuLNCiubi5v%E4b9dj+F&tuq<2FsNAf2>Gg#OYZG3hi;Z+t6y2ZS}ijIBU^8KOmkdbWbA;sG2<KS!pUqhcS?xXQkb_b?#KyZ#!1NgOi1dRc4|n#_?ViG_neN$@BQ{VA|M1)4iC->TNPs#~pQeX5yig-0>RPFz0Tpm4lMhbo`)<4wQi#+!c1k2i4hEOB6yOLsNWly?Nmn}`7Htm|cNBj5J<b>ie8pnwjvkV5qagM_sxb`Y1!K?HuRr70-$D@cA-lV2cy+vnif>EDY>w#RNi(Cr6F`$5uvkhC8Zw{I`!vSA0wu!CgSK{D(hY4#KUJHPR;gJjr2GVCB3c47N26r?0FQJXn0RqZdj^YZ!ih(K|ayCleR7SXk5Jss$H*Z8x>-#)<cHOM*6)_i%g=7NCG2rZEXdbb#XJ<TzZkBcX6ug%wk4jZ#p6q%LdvgF!!OFi~<KpHq3+YxEfsi*$PulA_H7f*@WeYGc?1}CNCH3zl;`@pN)NCG<3G8kvu8K-;R%|%{L+W)Y#SL8R$9T6PQ5PhH(6W_wsT2#Y}j|M&L4Ykth3RBY^oDaHT8d7>DkqJ0?^6{wM9QCZ^$+MY`zkRI#!g<&Z0DFu;3$qzbnMT)8?Ac6@tN3tF-W~Hs^jM8{aD0YjG)af-BerdIm_}rmn7Ert!lX!u+a5+p4H_TbTScKmGf^9^jvM;n^0=Yds1RSAeA9!tDX!km478gf9csiu0IvD?{LRa^@a5;wubyXOpsuI~Oik5%Y=2VHUL@v7n+v;-)FB;z7ezU)FA3L^4S5?)IR+_eB*V#-0Rz0nMemugt+@lmBo$(AEI59X$Go4uhjO`i!A5UMyHu3U&izFHz=5oIVZCGD5%@bgyKbUfHCog{P2XmkcybJ>J-B^@z}#V`9+<Q@9CW6uy-64*Vo(<mEm)~^H6BXiY&g+~2^=LJG0ho2R<`KAFQ(xpoKY4rsEB(%;s-nUloLK26v0FSsbs%rwkDo*+*m9)FcMh63Pj|@o$<~I&rBFICjp60V*2$8JT49GUK+hu2L>|@F6_q83_*JsO(m|&<WPe<S~ihwmN(oPixmys%2%Rl?9vJpxw?l$Bsl?tRNWA^ls-eB3(f9CG`pf30+%-Fl+<}P28tUc>1G?7Zt495#!DRM72(MzImpNV{^NW5y|#Ag&N3f1vH36*UA7;$bZuMe`bCFQ2gW@KCs8=u#Gb4QNuzB6IJ!W$JCU>mTAD`JV$=y48?kkW*V4gCM$zy>M1_#To@h<Je`^^-NrOGuICQl1i*7v4y*qCWO$Z!D-jR7?WKYmNc5(ezp>{#Co{QUk%^$d+XAextuu%IklYl>SbRpUj5ARm_9T7$QCcY7%eiL;s<ChEMzWJa^`d9&S5Vd$eA5G3dkvR;R!H@cI;?m6=Cv#MI<4z85cw8J{<|AUabMs^3w8wD6P2~=SlWyIYP5W(21v4PAlOhU5lMiDHrV0p$o8IL&+h=Rgc)N)F-0|0QE+y-jpbi2$o+d5Y2?u60`q&~I@QrhYU^+b^V3GBNaSU?=<54`AB9n;@MPdS-=x8HGk5DnE{4BAav5k|o!H(z(!ig?S<lU@|-i03bW+i!bU8M@bA<6Md0vsiAKP5zCiGv!QB;hta50U3i5PhC1`<d9AJ-$}$5+#+AXBeo^@$_W20EFzX@7_NHF)Szz9}`&gpj!uxt#223?a?gd1DX`F9YI6$7;pycgcLgc$)Njma4Ia7UOh&rm|>KnGiI<N#3!ZMnPrV=T_Y}-BTlVwmZh7K(iruX#Pqbov^13!kkV=+Xp&Y_<q;7mV0*@ha)b7e{2XemRRqBayZ)^Vz7s|y*_mvwlQtOMSrVsFG80VN0Y`7-*n*jm`IU@jhvg#ntIP3#m59Ssl5`w559Iq<hBG)6w1^n0G8*U2mkb_?bwe=@#Vby3a}v9pq6MgKVy{?d=b(Ci3b*na80k!+>!qm{ocOp2tbI<-1_4=e6r#*oaGSFtI&y;<JXaXHbLp6IsJ?R4NlOu*&zleSRzmTmI`G`J4h+dS1|gzAiJS?pF(uPf7L1RO$ihy9aRw>`xC^6l(sa;i_}ZI5l17<;bqk%E3YKRQ2+G)i7KLy&s9r~Erc<u%CUO2;Gm9JN;aEzGCbu^>ORI>A*l}xxQdlDPnsW~kn#zRUg-j;c%vcX&Kle&n;|%4pAQn*+i@}E&WJta<T9!7g*GVJWa5dL(_1iRD%Qam4HVxNv4cEU-!)v*Q*S<}|>$!&4zfHprat%ND>V~^vuQeM^b1iRtWy?$BAKwJ<uL<?ng!XHK`ZYoPn(%#1xV|P(UlW+G3CY)l;A?{KHNp0pFne8Cy(XYu0-#H-6q6O86?JvIRk>nJet;xG{^G2W%1%w5!!#Z%lp35at|Z<n108lz2W{o>iK8gYUDYR`teL2=?q{YzB13)TIFC^C9MSR^|H-|M`^e#~9pYpOQxh>)6QK*Wia}hXBSR}kZhA9N<T@80_&Z*c8Bybj&jMZi5?Iux!R@%|CLk-Kr-z3elNB-Q!$aQ<q;7?!zJapXrlzB*?+}4pfLRL{h0%nFo*RRGVUDJh3CLe!9zYxCa{zXMQRqQ0(V1Sa;UM2|U>bIHyawb819az=F*%06D7bYgp>`9D|D6}x1emX|nm9g#-A5y{W;@r|42h?o*$p&UxpGT(uCXQA<^$7w;D)I`t7zLw%%pPI;5R|?CSZG;G=)}b+CpD*J524xO`^4Dov_)$MqM(@7-r)AT&#P|=drl`)t<xQYaObEP@#IAMWTm1j*O179hqS9hAXLPuH@@wMh--Z<OsY?KIr+CF{`7|G6R5SH25Yq0BMVlV2i*3t4)#?sfH>ujv!A1W^-9azGA-&E$!;IPXDYWV|1bzbGBj?k46#xqAXOm(-t_Oa6xDkPC1hKmmAXJG?sS6elEW>QVd@#i`A{C%}A6Me|LLtKWP5)R{i#yR;cbJkz-TSYf00~p^())oP;u~E*$qnfgtgUiEBc0e#u;l)EcG;@YQC(qj?l35sg!5H`0aO?ft#|?fsWeB=sfY@=LyUailZxn2~c^4+_^hHJXjBp5`RTh2^&5>!&iW)Q;eb8-ILHr-R<3Lx!Z8bX|5Yq9(Fmd-nTMDqb^$hj{Lfd0}MVkO8*^_rWy!GffGTK?3nqT_PxP@|=BPdSZpp^8%#B5ift=>MWD_7|9K0L#os4dl8ZWC6*M{5DBS0=iC^}ON$i5t??CRCwpa_kmh(gF?pp~XXHXx$pBXSB8^$ia1)Q@Zv}F`*_1l<v5FDVc1Pi`t7hXy3IfTg8bs2Sm2q8qFn*P>am3V(viqe-9P9+uCS-RUPR_AsN7_P!sl$*~ABr(fzimN|&N2b8ezyJo^}pVhO#X0xNz8aw5g|@=QRMjFfn1m5`88>Nt(;Sj99))3KRw7$go&llfFOOZJOhP&FgQ%J1nWD;J!lb5GSPXF=)Tf0?HV8QY}COyC#K=l%}7Ymk?`ozu66kPBWxF~`DD}{Xgsi)?wTT`660L)?zB!rbvo*Xuz=`{;%Pz&eBAps!OaOOSL1Qm8MKD%6>p<=y4vVxdo~z$ee||ZBO(e!vpifkmUDnxm3OP#nzl-&1Z@Eg9K09L1)L1<-I3;H1{0IpOc5(XEuBNNj-qrX#^D5A@~P2)788V(%WwBrkhfZ&16j8HH`|=K!oG)E!(nEmdN5tMMw_t^@-mHxI_H+f5b$w|M%b0rIpHv+GYhB~`>37%p?c;6HXMz{9C1E5`XBdpUcb`2&aNr$q)iSC@yTY*l0Dv^_+l_)i8pm~kLmB*_r!#@y5gfk7166{Rqxy^3MVD?OJ}O8>jt_|ajQxMF1NH*!*TBo`s!##_~@1({3hMW;biLlLx2nytNyw3I=N+eAdj>S6E#{=x6}8hGnzUuTe#?iwA!Z7ZOi#l-o4*dm8)Zq$5+q-Lgk!kqGwzFB2EncwkxF~l<!}aLxj7w@NRP+Uon7l-6PWS6!e%$9TICJJE9lAy@6)x%1>1P7Loo?pNWJ_?)K5Rbw>9Kef2mg3_%mzq1DSb+fP$-!4q*bu(e9Du}XX|R*CPOY$m(`$Ge*3*v)+S01PWPi2H?;H*pRpV?9wfSzX^gd2wiTvpVDWDy#p6#2EHtqYt|5fA*OaGIqx(Zxi0%7*d+dK^5P&<QPWmTv3@JQ6RfZu!nZ88xPLqY0@6TIjwo>o9=(d)b-i)^E`=1)^X(&j&w`YKCCKi14<NUqlZT3oJx3T8$0a_l}JoOR2i}gacEddD-pI~-DFoxp80-^^p9G>N;j@v%2b~t1(#X7o0AkDjV4nf#B}0%pL3+886x$re4AFSw6kfp$z%18ErvmB%>*Kk|Du`gv+3!rL~1#BOSE)$#`?mH^I1imov8jtD*BtHqW_UV{yQX)|7#|I|2rjczg%K_RdViYdB9znnER{KwA(Ide#vz^OP{Om=UVdlss!ChLhb7r=lzZ>+4q@G*vvB~2`@v0Eb(^Wdx(+(>uUI;)WVh#i_ukq3e6<5M%*2-+pXMkbXRenXOvei_XHD*lw;I499^=VBb&9eG(0`gePPg<E3Vx(hA8iMviI9wc^(scze0uF8%}Id-tWl$hHweK74t!lib&3M4Y+$5Jgr(16#~n3Ribg7`MKTBrYQf~{NvY3ff{Wyf}(|hK44Y<L^p3ijDmuZb%MjB`CL78v^QsV%JR2<%G?R1a%SZz2Pc<yVOlF0_RY#B?Irvv(YQkW4p=z=N;e^Z2}SdMO6I~qhT7vT#g1ukf4!Nq3c|P}f|`ya$iQDImtU<0xs<8Psc!ua3EDp=kRjr^w42S8(I=>xgeB&n+u_kb>wN{3s1E1j%i2hCB3hfUZ!D+nW<;drVZDIqKtY$!5FTLCUdA6}o9buX1hamEnouHg?lc3!d3xvyOJtKrJx!yY1!CRR^b&fg@Tghm%)niI3JEiD<mE*J4;S)q!4IG0;S)d1<zZe-a^En~`jvUO^V(@JV~!fyE+VX0Hi`;!Xp%9Hz;GTE7cp4_FW;3*+}SfZ<MXNFmnIZnFHR`#9f2srW^D2X(df;qxxD)1yXnR+Pd2`;lZ_YV8eiAB#uI0R>n0jsmS+6xlZ=0h6yt`Iym5N*8z&dPd1~>mO)UOpS<3J)xBf#?$c*H#fXS9ilWvb~f2qEsg?XGUH-SkM9<j(v*e=3q40iXXEw&A(9#_%hb(?h8&#af+SeGo=HCDt3J}pI)aRB<b*ZiMfUcNPUQd$EhYUfy+OATUu(N0<sr;VM2Y`=TqbkeFMgpGLrTXYVu$d*NZw9~H=kt#zVwQvJt-&3NfjaTi8f2p9<l@z&1^o$E8+Ng#)(=-N=QFIXS^cKkVEKg7g9fk1#1{2!p23K@i^exAs(X>tC>`Y{iPe}GH&C%!!ne^eM;&~J9ecQN|nNLM#@%Bg${Bvp}l_&?Oi+@<%CCej|b}30At(fUJV`DYdA8~w7Vi4plfdR7<>;ou#fz8&W>Y>3jXNHaq2l<~K&T9#})Kvs-Zev^$L2D)$D}bQgE~3>~9`Br6@Suzn@@^r^1o)TAma`<^q|M316RGnJc&J_h59P2>{@<4@%u9AN;^RF35+?i$g2f6)4BvLu>bC}w;i%)&P4+LM!(P%c?5fi#$fMlLl^}h+vQb#tdA$<&Uq{kfR*`v{Oh{O<r|g7>DIxn!Vkj*kJC~Damyo0nz7<KDTY_nZ@O%KN+H9PAd*PVzX;P3SSeCDA5i!U<uFOiX*(Rh>O*w8=V^-#kC!==QJoENmzW&gB`ttvH{V9=VWt`?_GhB>&-fDl!9<#Ht@B7Qmn}wd);;WJUi49AfT-6c{WE=J>P>gJ<4nitU*)v->axL;X9<}42j5}p)=|>8!NsARHc@0f&8yF+@5|l`Y^}e>4Ivyr(J-+tYJvzmx8K^vFT~V)rACKtQ5TJQ{?Dpx<C5-zP9xe%wDy^b3&1!=<Yiko7WI;*3i@TOC({NfYG?+a`J4CTKWxp)hzl;|}@loAzk7`bgRLr>+Go}UrtHO3_HQhnP2`CI}y0j4Zn?wJ5V~eFYVdLB3gv%M#jmsF-jVz;jeOC1{O!)Qe>bGH7ZD74I)A~Eu*5AXpem(2z#uh6?av6RoBlwP|Fuxu35=?9GtFkO}VJ&H7_{7v{om19QX6NPek}9@gE?}Is4NW{A5(y|KsjhkQbNSQJ1~J-<<}O>&eZGN7>GKR@bLQBF?sCea=h|?TqY>>A+C${3=Sld49xUt-JK@mdd@KA9!Qbj*l|FwqRFzngu<cxL&|&L5EHKtI$v-oQ^a6;9Sa>!YoVSJ{o0;gaS71xT=1y)m@TkP6%wfJ2k0dN~YdS#qDaEg|1XmA;7r{_|)~+aQiTvc3z5N|f_~{cXFWnM*P6^*OCRQif;xSzdzlXRKCy51Qy5Q;Z?1M*4YeXC+H+@Es3tnbVI~F@%z2}IO<ELeeE@d?3j19ay&(%T=3A};G(|+e*zF~lBHTOKzS%c3Zm3)M&o@?+#tX6g|ZN;>(qg&W<;<<dZ!fqe{t_wD1SrKLx!9JXY<zS>Vpv~-B&9^fvL?!hS&378bTksL5<fHdIvD1y%3>}DvK0Qv!9IysBmufu1n6=I7ViTurH1Q%Bhkx`IplUuo_W{g0-ZR5EU9z)bi|^6)XnLd~SUt5zGuXDpLZY!zY%DRd59nY=dw$-FiU|knQ0wyyJfVMne6Ar=f}HS>)1_YUO5>O9?7slR{M<y?qn`7W$hq{(7dNEalTW5^p7cFsUBSAUbBN;hXz$Kxt4y0($<DUxg6F877^mw;9pg7cK;@0yI|fBH3T)duV#V-0F=b~9<isBM`L{6$rgk>f`@L0hcE{$*O#+t=W&n9g;OpQ>vigb*gFi&PBOVnhsp{JLY7#=--%JR4<j^B7gv)5$r;xWUBx(h;Ap0qD<UeXby(+6cc3j-j;o9lFwyP`M;;cjysR@E!Lg}9|veXgbHi?^|T9W1Epbu@Stuw8w`DoIO57X>lm~N5*46zlcHXBE>NwD2Q{}54OiGfYsD+!4&2y4?lD83Rj$)94)C1W%tG5JOUrZM;zVxbo@O&AHvTJLbOFvZEjI=WY+3Zr$)Wef8Y6J-2>oiHRwW%bcoj8>;J!wE(70&8K|%$oX5Q==&)DtvnqtfNaWn<mZV$*~inLAHsgQWC)=_ES@8Gn|C*@N`0)`>N03X>q;vK}_5StH}1!A4iiDc2D4g-M(N%zbcyz8$AXCEy48?xI?HvzEnFiC`zaE5TW_!Y^@!q!Mq#`bCw3S(5JX`8j%U%k#4w19XHsP0UOquT=<FWt#*~5e`tr=go-9IXM8jBExRVp&-dQ`s%lpmr!GX!4(Vik%MAodEkMLchoG^%zATm1i1PT%G*}kY&3b3aj$$M1UG<B4{#9+I6KPyNa>CZ!8+$z1Cqvv|bnpz$c2U?DD2LPmGv61n!Uo_ZyeS>Bf_C1$eD>rSHTAF>tT&2Q3zt`dG=7Q`Yh79l9S^Rc)nBm(dYJcAz4Fj$8;K?&#sH(~rat046$XBN(lf2@d~EJY^dkC&0oUl)ul#PEf8iZCWZ4KU;?o8;1h0%ET$_EaWGaBlK)TP{@M6>TV%Xs4C%5<dm;F696t6T20jxJz(4=ISfx`uTdvwDx%YfUhpxND*zkak=Z7rxa+-7mvN{gO&&L{gq$NNAwTS9wk4-f<*vU(Cu(R7BTGt46x^>k`TN^mb7llYvvpJ@)YbqTJ|V_RYFXgj`)!Qt@}A=ZzZv_g4J7f0o53$>*Qh?LFM1AX>p`&Yx@ns?*&C8LMB?UFHZ<YkPA*vo5;C@1np3y-~P4fL{A#wD7c2u2UyDV(gDI0iA}d~>pirz?ZUH&g?Vub~#aNeFkc^H5l&5`*~K8lgD^GKjcUpl&G-pF}uPp}%&l9<G<mbPh{uN}DyUmd6(aW9%lQPNESN(0FcJt8(b)4qtN!anxFkL%K;guL~!Ci<$VyBKvI>AUkhRTcdyKbhLvty6J-lbhLvFdRMJf@XrHbYPfG|y*+B4K6to+8Jq_j)rInm+fF-13XMv+PIcQchc&H~AHujW21=oa;K3a~I|?GJ&`(gIf&tP42gv;!Afs<k$2+V-@StuTq(l{Ku|fnx<5uTn!%Hb<?{3FG*wDFM3hGP@KUC!9paX(Yq2FMrpQ}y04czeVVoNzrs|ONez@)B$Re(<((LCAK=B#ELz7%1(Cp&Jb9j8>mZ1^_W<}6rdDWo?7PyV)(|F%QJb&;|jVC697!T804LkJvc7tO1kY?+;;jK<=8Rz$(QsRDrW8W8&B*g%j0*jG9HC8MQrumC>7oRLoK&t`I)dIiL`nWKp2qKiW4%^CY>EVH2)17yJQ{j*=*KP7saE$qNun4`^3*Rc?ypckf-)?mmEu$pGmCZu7=b71n&1@`CL`!9Crh$c-hbpFNG6h?GizFOKX4)g=x2<;FHn^U4uSSA(|uLV8~Q~|gaSnV_F$WG(kdwUKNip6yf*V^RRin^kyR`L1}+KbIVCuWS<UY_p0d`q`e(vKl`HE}E=BC0TmkAz%pA-6d}E~6%KLQPhzH=M;sgyq_UX~nF12~UARsoCy#pS)DH>`Q_?YeXl^sv);48{9HkCsE*ZMl-JtM`Q6Pm9==g6%g_zRCeB?_Yi*3DnDFLY0pjd<N;9!{?%qUG%HpnJI-`gT3`vXDKJhuATTcOZ1f0J;yA68bOnane5$TApZ(sOH`_~6-Q&}v;HK&BD@nflu74mTQYOQ(o$B7nbnhg(zh<8Mt25Pa<P+6rveR>3cqc#o7fDcGu~Wj|&N~Qfs@q$=+g>+K3!gj&?qk(`tQ|sTc{|A!o}>y-GKEW`@KP>6_)WIrzi20dt9N3>-H8=<Csy2@So!*$_;T5?ss81UO|X9DZqa|)z50yJy7LgFxM+a=1L^oZk5cj&xOD@!Zs67p{;tRH8}ID*JetX4;MNV?x`A6a_-@C#_Lo1_)%uM$UodcH>p%0}-w9AUK=ELW0b>2nd(@N1s#~|})~&jAtAAHC@drT4pA~f^k5#vB)va4~>sG%T+NuBL(N3-U4S-2sI9X8o%&>GPK#c%Zf`<UGYURHZ6(x@~w{Fd?TXXBy{;p{3KLmOIji@bothse-Zrz$&xAxu8;@V#xE!HdF2{`oyn+46!44ZeV+&b!1YYedUAE+{mDwD^$Tet4kt-E#We^)g7CxYZ3sMkfk$z$EETX*Z$-MaPfhPK!L`e=LO4+M5iP-H><GsF6wD(X~Gr^>R@>c$UL{w+~?^0?;KU32TMxpmk6w%5R)`(*sM8ou4tkUXxrb=TawYi`}O?{=*`_{(1_YwQ2v!=nkP?9utmXXj28b!w<nW8r_T`s1qkvs^XF<GNdS-L1Rs)?NR*UQ>U;llJ54`g2@e$>X|PcipYK?$%xZZr9$%U;f%#um1@Tqb3-$NAWYC#XB|Bsi97dm07j*A6Maj!&R6(K5*+kaO*yB>pu9qUZejdC-=wI`q#f&lg9^c-3M;n2X5U5UwJLV+hIO(&81^8qBh=Fk|v3*olVqU<3A?;pVOnsXx2ZnD(O@w`){^&hKveF6NX_n=UaxG9~)U`8#Jz(><%auq$!j3nSK5^QKq(&F6m54up5rqyCL=RG>OsJHk;?Fy`Rn0kRVHPE@}6|`O#>ob3M_(B5fYNkk-+_J8dD)?u9KK7^m1ge)r}3y?t%ao+O1D^>o~TsLxJ*?A5xNxHe9Y2s_dkkY~tBdm~dcZn#*u`5@+id7UG%<UVCzzLGIO7wc}Z(&re??N;YRtt8Hn;yO8QUBDGng^TehmXu*hJ~_^|6Lrh#J-Z>mQa;qdDD-G%?NynEcnfT*gLqt0{lJ$gAM_Dmd+aMO9wWR)zp{ypJ(G~9jHqg5y_!25wmM0d!{3!B=JeLQlT1e1Uj8Q&b;vBEU%rmRMH=tIcwF>#EXaP44RSbxle6dbbqJxyk5!%hU+N9*e3uN*B<<oQ#|Co5!z1lyU|cQ5hcR{SaQrat-i23|2}FWU@2X(UM{)~(LxXutV?{_m`ngFzGY)5o9lrQBEKkv=Jr={^Xs)~{47<krG;w0_3)`I=H1T^vN8TtZq~oJu$A&}z0WTYLQr5_~a_#YE$Ku2p8P}&YEdN*{3uVF&slfJ+!d}2ZivH^GL&iPN3DT%zQ1&Mr3x3=_#1Y86<M{A64B9J6KoH;u(CY_&Hdx<`6TPs<Pn~vpO_q#KZdE(svm3uazxLc;p1y-a!Ryw><@rv`iE!uJapC;49%6`h#NiUeRg8}UK^`2&O7u@Z#=nv8M0h}rEj~_M7sWCD1V2dhFb7NILO>v4DAmT1)1m6<2?rt{EdtTIW++L^R??Y8si^01AY^(e&QNx+T24_a9^8KFx{}0BWxwAQh3Ei)m47pJRjDA#Q9n)^ay(sl%hT$+Ts@QHA(K{Rg76x=J9ca(&(l%(lEJUgAKygfiY1tvDV|NllBIx8wSRG0f)kbr)rXyFbGopT{FU45b=54XgFA?_dMD6H|3EjybgrATS`{uizNRJ8KAG@H69bxv7q6=?ZrSeF=yWpGj#7NJ$GG<CqFk@9N+<?|j7x}&B)^`yr#UdF26O<<h?5(@YiuSd?9%QKVT>AnT~*@|krEfwUp&=W6@k#Es~T&M;8Q+m^Sv$P0$-FJ8t)MelN#`jIQB|3yvT!R$T8#eP7#jLj}qpM(|6|cdoZ&?qvAW|*f<nQ8-UP5oSG1E=YIPQS&b0t;t@BU7F)53`8r<E(+}_7{qh>Ie-=S?w_^WqHXIhig`ua)?b+}og@xe^USDO2!B959Lmx(`<6$@@h+AxKwlAh(Tz*}p*PuU|yi5W+2X}zzz&IV;so__>m?2SF@Y3Oo=dlyc*w;iW9hE%n$%`C-QgK@2EM!`#Wxnx3+$K<Y(cJs6yZiE2b-QpKq{uJtp1t3G`&8ZjW0!>c-hTf)yC7E98eqhgl{sT=gA_mO`?Db!a7zUZC(;=^i>H2;o9w2EHI#IZIIMDsdzt~*v<P!U<{NTdN)9*|$8^x2A?*fHb&G8ii;MV(`D8Gih7tRWUpT2$g_Yo(#YuKfkfc7c1uu4zcwR{xhT1p$L6YEs7VvFzOjuyS*_C#R!cDyY<s%|8u+#kt$6EOx`YAUYRYfEj3Fgw{M3+EphziR7t_WOGNLekuhO*L$E3y=op_(M_P3qn%HvxRwP0Wf1)EKxnCu1#d7;VX*I}Gz)w6{zq;_JB5S`_;tm)n~L!EzY|xktrZJ<)CN?LMt9lGYYv_AVd+B}%A8{$lO8LN>hUA4-X4c5PwO8q49<L1eNC)i`+8Qu1~2mm^=T@7bj2LIr8!bg{#<=|YETb#hD6O5kK}sf|mTJI1Wh=6@q?TTH<IT1dc3yqN{fU||T@s*nUO4Z9?NVm^bSGRx2ODI!SQ!3Ky_wbFPcyI=(*p-O}nQ4Ntw-1wXSpzTLQdBYF?N{g}KFH6d}5Ui%NGGbJ2Rz^Y@bX8fQHaR%9jysSEdTfG4svUjMY1t*Au>@xBjE1w*DCTGo^r>7{E7i3%xyq`pd<lW=>S>sq*cV~Wd@`CcRy?&<{6_PYXg8Rz#RDK}N71yeWKDi$)?^VcOFsva#!1)7+s+|^W`7)u!NbU5{_jA&71C?hB+wQbi7#RuOXt^wIy-J#pik?2Pw8bOKys#7NU;@6TS&A;N}~M=a9L;)ZLyq0v#wrq8+(#~TkdERze)HbO;nYb<0`q-)TVPSay$zy^pber7V(3Qi3}VdWdqo~1&0q$A`3}qiec-5ZeA@Qm0;nfxTagsB9@rxDfFjHNeV7a%oU5OWU9+WU{_~$jtV)tw90$o|A5pYL>vSzd5@sAxNrYmt#1<B+v=E5ni4M(16LAFQ<ELx4dUn`qSw;_;!<bg={nIW-<Kr5qn<`mNe9?H3zim^D3)#oLya68Dg+!ofe(jy<M=ILoC(3^gSZIfAdrhN2Maff3!%!G-*>|?F(V;$@?z(WL}OIb$zaSp1XYnvnTU<T#T2g~PaT0-AJIe@7*0n-(ze?8mnnObyiW@AFc@oJsMFR3&0x^N*$c|G)dy<EYy$^wN|c!<Mi7Tb+rX)N@s!Rg#)~kIM9s~j6CR;XlBfrJJKdXvZFs(;WA$-R95>$zXXQY~6_9CX0smOpqAKPtmQ1nd2hVqOS_VlBtGe>$nn@;~@8~R(`4nG>#*9wYiNQ5pq@3JR(QG&zPbY4YKn&6*6ZC~;Zg-JSJfaqTgwSskL~EaL9+Fj+5CkPRF@;MSOqS<*;84|^6vqvHWXCbrQ^`eAGud@PYc3^HI0Q2N56FQUUW42%;u!NZET((#264b7JtH#!zRuP_MefpwoT%#OGMdX1V5cLq21W}^2joC!>p)hBLrr;^ftoz?g~<VK3MST;V+E-fy^!ZR2PxWhvs7ksRPvc+bnPU20wULPJdPQMv_6|w+C{+^jO$sYgGfpe1zof3`YQJm{@^&lOqn!BXF87&9jX+1GT@qp*W)|^9_myMXF*B1)s~PXH0H9NS-C%+86t$PvNB@EWeFgNt1OcCSW^TcLZBB?0+?Ia$X@dtsdX6k=pLDjPF)-rhy7E6o{~hB7bZW!ruySgM%gBJvyK;Kf<^0HiMl!}KD37DHC4JEwa!C{&1Pt<E)fhT$dMWF0ptad$c<b95OAD`cWBmJwu%%9-L54R5*Xs^hWXB0I+s*JgNQ!3C)Bp-Jk@`eM6AmV&Q6i$HA!^KI#C@Tm$kBKs90m6){o!O-9+Hb2W5b+kVJqZViJUKnD0yw{_BL^2ZchNH5p=ZG>W`%vezBUp9ySA1n-q>vVpFH`ZW`@(FLDLE4DPNSSDZ4v70a`lP@TfEr@n$Z$5bb`U8aeEtW0`X}4pC*UyrKJDMo;ZK*vSxz*Fw@n|AESvIK=%hPFLOqQI|C6~go3`W_kC|Myzn@iHzmt7V3fRO;@f2*{c(Ns@~Q=8bCO_t^USeozTVR62bZvEhhc9Q<!4#UsoxlMA}jvf|fJ9=nmJ92WP09>w-qbQ#=lQShPY=#P#GFDqMu_u=@ElK89)_cNXE0a&_bzK;ng(dcvWKLh2Yw6rdq+~H%b_98X{|hY#j)oU<#R1wu^)n^{WXESSj>11+9D15Yx$X_3)=&i62A#*DH0Kcq5g*NZJ;*BW8+_~+CZP@1lUzQW7_5?ms>_bWPVQLP<%-h)32CGA{77;cLhZ%bfbwNIiOe@UKRn^DIW=>Z@coW+b4B)?<H8g?nVvau(7f_ht?Hya*_M%YXdV>iule$OZ!->Pd5Q8u-=y@iD{IeUux^=e{jH45{8GN0RS{e2<M#V^&6iJq1<?r*!>z`j<W`%H&GTJ^@16sD35_09ALLoI4?C|-`o%<I?FwOOtSpD6lG?_A-y~)RACPR*m>De5aKNxMsBT!YuNoA5{X!LowK#!ws~Sqm(<ua4m2<8sIy-o}n_wtT>1mgcZYaQo73<Y#qJD}^ms%8AmShq%cadqj64F`w9-<&aLP{&VNd)1r2OQ^$SE=%SRZY?hVq19czn+B7OG!QIj5aJVGR2^PoK6<mun;K_ei6yXJ;~x@;&u~EgUDGyP0m>=5z6O%7pC~^Jl##Qvg+`$h}O$M7C>9i(Rl-r99f}awFw^pQ+Ow_$Kw@Cd+?s_CfdKE${x?Frq!-OD$~|>!Wfo*Y>kHYA&B9;4v~<&gHe<a<x&+qgJsZmE6L{S7`2s{hz<!!UKj||mVONb0{P|yCMqmv1pO^MAfXArtoBh>Ohk`o@${C(P%@~tz^{a>M82;QO6rHx^kmQtDc7PUOcKc(QKAtuEq=q?v$9%__2^(C9mV;RShdsPelzIp8C5>@6hXLuV>q=(Q=~<p9u#uI6NWN<6zUGX7I9_9X^}+k5EJ&DAfZuUxR3oS)8L!5kVT7#0U4RSj7r_!qG3^UJf>?cDs-HH;DdP|vJI*$=Nf1qSa;nl@7J!1l`@!5%$C!2!MZUsh3|f5qur5kjyZdo^iVS0YZkHB)_4pV5fVI>&<|OJ=DT?r=2J-$P#}HF=>tnldTs&RYg|m%vu<~+g~+fv*O>)=(!yiTV#Tatn)!~-;-_#j5^Rx3ArgZP5evs1@Qfm5PUOoveG!_L9_5U;!=r)D8j}b}Fp5<8<mJS~m}e3=y9?8>6tTbv8TQJ(*>K1LY~5i(utw90kgWLe2yMyF6$_i_M91d`)$5CNt2t2}x60qB6W^9Lc9x18ll;j&X0UV<z9{^zqQ!4Anv~wor*gi05ZB2wUBqz`d0ywN0$2!8j{NsR|NW%@&cCF!ZL!QFqI`8~@Z*vQUi7#?_&daN4g=p~lA%%{7WrCC@~0r1w8#;sO-m1v{E2P!4`P@+P!*#5*c5)|!_esM{=PZvn^yKKttgUPn5M<f7pTUTTSqwQqh1dV1e{+W$C`*N4B_Cf#iP(*Z)RR4?a$;-ti1%{K>vjh@Rqvf|2T1G9G(?$LYHzoMvAA^3oiaGVudm#E@z7}qQKwJB_;L)C&7A$^A*i-mzIIMg}@z+TW2#U&on`b{B1<IBI6aIUV?||U_4|;pbOx5_43X3Q|0Zx*;eDhIP~*9vG%>m_B$Di$pWLQ^d3}Q%kfZ#W~>550M3~kczCOPA8XMhs7iekZL0Bff#h(keM~L};lNNhE^*d`nlBfRo0q}P$gpWS9rb3Bkl#8rBd5L4X&6*{)&+^{JzU2Hw^pwkwY_%a!FaSuUYg)R)cG=eLL05<g4YtqLUl(}0E$y2=<y`SLy{bFKIn$4Z_}I*ddO8hi!>2OPSd%UW+i*R=NPTEdhqLezQ@b1mCCq)*|w{pQ<gCSf)Oy8B;A6m5p@Kv<d~4ikVX1>x>RGe!Y-h686)vqE0WB3k~$N)S`+JWjjc9KHnB7p9h&(GHW0lS%Zwq=H`)fMMZzJQh0|(jE`(U;i%r?5t|=FaJr@J8cCD%YJE*&o?65r>q^?2MpuPXy0OErbkWNaRn$Nm`&lcp1ZiNZCChB|o81Vg3;`C=03drEJai@}+9x_Rmq-9G|rN%n7BXW33>Rp@)$8s>`aM{I;vl4cO(~B|j8C-aClQS`8%>vn=m~f!FqmO7w3|OpJze=4#kBnXIk&&7vd8m0mF<B<8MYy__n7Pqc$22jw8W7@iDtR00rGqfWQ_BYNs~JRn`5@A5gppjg!dI?->08aRRgtGN+_C4GZio5cuBL<~4<;jv`YMj6J}ar2h1Uz`1WiK-ima4^sck`L4inRRSLvhKn2+ucEnCO6Ax<fNYL`G5qH|SaRnyo;@UWV+$*LQX#$Qfz4*NoM(qHz&TUi<Qvm#(KFBGNO<xSl%PJE=%+R|Zev*yT#Z?HBV>T~8~!|M1;7mMY&H(#paF5I<Cv(7$y|Jt#o81@|zMq&Y(%LthRF9?so*M!~8)D4rpPKjreYlg`P4|ycv{~9N@VN%WQFa<+H0Zziuam?k5pf{RLtjiO^r2z{%i9BctBy6dg)pT;%%w*%s3-xdQd!he+(tqdlmzJNu<L`xSrQs6QELvD0I(X*Va%5~EN9X0XmP_2JG?rD$TB19<!;~JA^_&PvtSN&6r!>dT!*S&J!rT9k-2Sr^Ceuj@1Xp%v+0ydOnm|U9wmxy-u=AE~X10?y`BrAw1t+~&*Hb4HOB98x@Nx9Ou;?D*PV=O!&$=ItF0CY3oKRL|s~YT!M`4Xa+nC_*BRz`a<7FmE$GFmtU<kO!1|5wA><uWh08<Y@BW<>3<MHL%On@4OAIGsc2>%T1M&P@t-vs*2Me?Sq-+ZDs_&aYD9h6B0ye<7l)c?e_oI0y}#I3jm9H0gDCFp?#CtY=LjP%PUuP|{JD;uok!~~C345LRPuJKpPYTmtn10N6kWOARWBS~3~Hhwbg529PSJh9LUFKAFE1+eBI22{K~q1EiLgqvXRs5Lx?ROiK;53j4L{G!r)Q6T^+zX+Ny0vbq_DKstn7Rn17#kc12&4}l+8;^LQ={Dibz~2iZquY)JwRimgw|C{wZ5vtocl{OEY>gMRDVd~XTOnnoB*)UEB0H`~Gnpxum&F5;Y4ZvJ9!f_4`@ZkJ*B1aDc4n$}w<cAw2%yp3=;QS}zsKJFpwP~Tx~$H-sn|*Hv!qS^7{%$b8r6uMsI87_N40tM4^;gR82vaXW6qBE$86PR!RrZ?eD<nV!~a1`#C_8Sp-y~w|LGI(B@lT^L)I9^m|<#}c6qK)-4^;7V+nAf^>sG<IvZ>1JoM{4%-8wgX=2)}+>Fb)fiK;iAbrBCK6=EILPP5DDu`hq#w#op)C$ce!%g+8_zuQSc@G_my{P*a4gUhckE?z#US0@dc|+PxI8oK&v{Z{jUo8&w7Tv_EA2Oct(5PrV+NC_px+xO6>K|ov6a2wR{nWpU;nz_w&I=2Hr=Hhf=)30gB|^Yq4l?Pgo4mzUaD*F$J7d1&WR++cG#boU<VmRB?T3=lDD>O9)%BCxWCrrMoXkE~js96a#mnue?2->D5wl*q86opyG)%Ho&e~rt)ii6UA#%(o{R0+epBC)CV)W{e0FY%v@zH}q!D`smsOBuo>Qd4~65D6{^*>eiHf$sB3xHoCW{iq+3GmdY(|(D|fgNHKk@DmTNldI|Y>6}WT@t5Rzu6=bowOlj{{B$e9Qwu<mK1_jy@iXDD6NAkK#cV&yzS72bTzugTRU|wtzW!$@KO-g1Lr~Zqpb<ueTk&MM+eVtE6xhU^!P=~7<;TnnU}S{H##oxLhxZUsJ@$I%Y+!|S2t(Su=e|abs0~VRh``d`~jgzip_GYIK#tM)6gz<p-nOy`w_AQgyw?C9VTO(Ro1%CdW!4C5zx6*cl~>a%JtxzP;nr#Z<A=UzF~Bvab~cT>?N?2$|$2N(Fk8qc%@Ja4{na_>lzdzvr+=SQ(~I0`Wa;_F6pkb(}(u)`m^d@E>>-gAze+yPzj5Y`FD~SxA`Ab7;T2{(w^aykr<h}SU?$L;E--KJjd4*agu56q0ACp61m78Ghz^o%N%AZrAETSdfK*{wp(wi5zlBPu*(F)Rp+lntSNn+CF<g|Y{;XS5ouEmhJkbREYa9cv%|jLnTn_RYfKdm$*5{A4%SN~&b3E8_&mc;_6d6X*T232#uGfDaXLZM)`?!7U0sQ1HB4}+D!!e{+fy-%&bwCG$!cI6tEc_nCCOEdhkPShXKy?ge+=`F#vju^P+8v9TP@!|10l9_H!;L>Q;e?ogK&U8>$?;24rlL6*cVv;pRBT3M>J`O5=x4Zy`3<g`rSD=vwH4cc|0|~2k3{L5UJ*lj~}Xob}7AaN@egp;+jaifWI6O%<5~=oh6;mE$@6?7j|58VaL3sCCmXkg^x}&ypKp>NAWhCI0DwSB9_K*Vg}L7kf<D-SFUUP7SRVKElPZ7#q5Jp#fC6_C)v=wkhP|GW@VlB<Tk#pvQco(5M5p*Tcr@8?t3m6BK0BfEuNB`j{M|-{kjJlP;VN8;t3k88(Jef5rmS>pq605?g#eApB=5L5mwds163V{RUQ6;s*b{{j{ZPZ&%&yn{eh~UhgCiQ1692Ut9tQuRS~K0%!gT6*YVfXC3}_j<CB->d^JbvT3-4=!fexQU-WE9Gw-2QG`;W&hIUiGJb_?Qb3BMV@vgLYZCn-`6q(VW{juz!LHs$d>}l-hVPn4w-X>$Nu*dPPhqR7yeL@@e@VUS_35be%2CU$PPyf~r8x;lue162WXf$j(?`l9S&ChTM;H{4X+I+J)Bfc;lMmGCl-&fZ}3_uiwDZ4o4VP`(>ecbK9<q?R3kejjU5S#_G*AvhP`f+ajJDe6ml@N^B#5xE=GkEmiAN5kG7roxnAr0@It0x?_^TgL5F@>WEjB8aBjl+tcZ*BSWH80e|7mm%6k}i$Q`!p_gik@T>2>_9DbagJ5=eiEnpqj~2n@-XJPteMElTNz|0jlvnjN!#)+5EBS3>EiswG)UVDrtAJxFl9;tie(MC~@8bIbVy%SoH>YL_VC(68>SV)5D2O_eW@!F{d~K8GwwH5ud~lRhJI1C|)wUjXGb_Tl=Wul)bmTqi4iU9x*+7LRIs5L((au$=C^`O5amCj!dMUb5H2HB}Y7Wh%X<CZV|l@y}>R8j0UfT;Y2uKgF#7^)6$5)g==WH!g3#`K8{y(C+reljfmu=l^pWq+Kt7OXxk*Q`d<EAD&MM%?4uker~OJ<jb`$AH8%AZzU{7NSSzzmuB+T6niI0+zN|-|<gO&-oiT8FU^*@#qk!=ke+tK`xpM<?Pi%JiW_aJMgBpqj3~&EO@$L^XA6F5Z5<plhRh9(B9i$m}@Ebrqi>DV(*?6*MzS5mx6Us3Xe2R)ushFFx8g!&;YjFyMSBAhZzckdgP(yu74Gmkv;RiR|!j=mC5&T-AJBt4z&L18x>q1Y3{oT&nR(7Fa%CeSh4YMUu-4b830YWjHd498UOPsb=NocPqA&M1o@bwDCsgvd1t`LD<P1Kvn&?lKxz?ybucYsFj2|>LbP9`(UTRg@0pHL5#DEG|K6cd3;1eO;K8Ht!fkns=gFnXD)vEqJ@Xz``Ifm--y%Mt9J;w@f6(%NYYbH5Nb-V3TxT*obB2BIn6-rw(e)VPObs<zQGe@Fgt{=?;VJpwUKTlENqFxUBQafe@+Esjlhtrs~glzBHbBK}{_+gLR8=ZrXDcmHR(Pk+vGPUP2oyux2^2B^I81z%qv8nm108oBLcO8R7<#o~ZHn{)yG3<3!`)VLGSe*HWckMC8Tma^A&I&t_m?7}rjJfu-9PU|~y^TXS}>!P9FoI?HDZj<OgHIQ+{r{1fzGJb<Y`nUq8GH}mhG!oIkM<Y?NcgE3i#2}}saj-+wI{i;fu>CHJQ0X@+nq!*o3JSJ`nzM`nWw?<AFl`t%gr&h)&pd)F-yRP<!69GRb0Bb20t_0DnlenSCA5Y;S`YzW)bn$R`KB4&kNWqt{%EzvV`}wM$4wJO=<EnAd-R^(qAt<lqNJ6U?moLN;e8kZxVWfEPG~eB<)6m#7JUd^6Zat9F*Kz&o22Q!a-KMfhQp@7wKI_+NkhTA&b_52IObN#%G8MOdy9#u&|xYuCkQ!jPaRv(Qc6;i06N?zP#_3it%8K{YLnx{dVF?Vqw_Rj=Pagm=QZI#d*>DekO@F<dP~=AzbNrB_on0G-0mgpOXoq}1NouH9=08o?^-igamZ!G$v)-1>`$@1i>bz}oSSK(Zy>7WR9f;WT}Zg}$lm1-!q$8B?&1RQw@i0zw3B$*Ti2#*x{N{PD)eH@{5)qRwDz36BA2PR@@0WkOR!)9ZR%;vWU#7`Y4|HHd}f1s4g9leFEdCKDhi49X}nuYbZlYB^6t=847tt@4IH)~{!bk;?86?5`_?jJ++EpIg|>{)(mCK;K2G58(--khy>5+jG`wM1VcOv>$wgcM0M!O>*Z5mvZq;jubsC*%llBO<(nelNtyd}W7V}w*wd{Gn`PiG5D~J}w@Z4elz+5PT+`;_$kqt4k;bp^b;)ThyVO`E2t#RMO8B24(<z}5_W&boyrwSXOXMK4X4864QR@=I=8(MLO3ng4~eDZOX3Q&u6(Td>lJM91*YnZT^kSEhEf*cQ`tcMur46D&@i>jq9>hpFxetUCO1-mqwB<)*TBmG#|k7z0cM>&vE_s9G<34&{ZQ7J=p%5M=<hmDhmj+pK-gW7ljcCOxF1PT1v5_yit-=AN`HFL(0m~}@VHR2=9ftblyG-ZRp&+AUz!3Ab7@Dli(%*M$Oj_31iI>+&#V6;_;{gWAoCM3qQzc&Nk8lOPc2@^9${6xzhYvMhRF3>K`W`pjW;U@>FE+ux~hev#2AK=8N&j1ATY%)AR6J_y)0Tni=Pjs5Ry%Esp_Qzy6O=d<1x)YVE_@-~;8W?Qld_(pm!%dh06)QMQQ}Jbe*t4#ZAtdux)Z!aOz3;g=yS}Nne|o#Oo{JQ^noZ`tk5siH=MTigTlX`unVbPfA99qnvX)ktAZ)U-%#-2J!01_YC~aY1D;agt&&0M+D=im*!GS`9G@Y9qGwL;B7b&8%#D!$o-xu|3k5=rU$LJ;5$Na?6sejlg@+re{5aMKJ{`%e34>#8Ioy<&MX_v7lDnhiQ{ZmmkLwl{yV>%*+G2|_{#!-R67>J-ag;dqe5`&aZjgN<3S5#+v+Z$)^hvN?}<htm*>(GQX#B+cM&*{M%_+s<blb7BH`=1OI9-h;;w>%Ck37m~Eb^})<_E%!@<E97MWcs;1=)((T0LgoLFr0`ck;pX(p~OZd829PHn7H?fsoZ)k<)R{{2@V8QKLVX;cfbJB7ss{Lt^PN!MoD0u<oa3M#T+tn<!u^sO`;1>Z=$n)KYpZxntXudIh{;<TCn1qK#HTxS`18~SnTY?Jc+J<Y~M&LPx<#|FBH&o`g)Ly!v_C7d6`$vF8KS?kGI42d|LAL+>dGm;|+TUF${CRqWz=UUbG5k*D=IGTkYC94~sb3<Lf%}?~|^u*JmNqZEyRg&LUFJ?6STqrl{Zeq81>9QeQRzi^2xKsBe0q1YES_gN+#)+n<T0RJ)Jfc+257S-rPKSm7d>Fyy)!)whIE4|KcIQn0Tv?04B#W&hF`!@>LP^pwZEef862OGr~p-5yAuw0#ZYYWh(ToGZ*~8i;WkPO{Ya4&YM;pfc#p7@B6;L+!Q{T(@B8t2qV8zWDN(Ekt~yQqPO}>$9J){=VmVCxT#Tto_{tJcrSyyvFH6Jg+*XyMqa-;E)~;0Yn{%t{+X{M+YOJBzB;e2AK&tG^tpSq(J~?yu1MSto3kF{;X*o@DT+b5xgk#HC~858~nJ(E@FvQmLo@(-AuhcX6Py*zqx100yLU7><Nl}S4KpKW3czc_9u(Flm$<*@dcZqpyp(2p@g2NYgxQ}T?6LDs}O_WTnax|y=+BgFN;K1Qo)jQ3DNs+f7w{U+n+5SZ0`NptyCZ!o^cgJb@_sc?WsLb%Frnc_%^?3!_yU^yBx2=90fX<82^w;^Jz`<86J7&Bue<ilp}8x{ex&cJA@>yuv$B}Wp&rTvhIMwa3jx;gwDJMLtyy<r5mg(EZcO-4wj;kZan@=&*rtg_6B>zHIwj5EN^=<D<M0hFl|h3sRv9OSdah66QO@^gfH6&IXTF2!{I3nRNh)$@`Q}*VXC8?mR-57QKf{P$Uh5s3SW5(_|z!IJ5kmd45Qw#GZ;zoNI*bAaT~7NF9kL)`9P1U*o<zzsIM=*4&FRBP-PqlGa&h_NoA8vGUMuAQxbUD4>rlsGOwEh;dbCxjI;o~ZYA_migFC@0&>To>?z2LFeKqRD(lF&(19;$g>87W9L8xpVq0*$@Hs0nA?5~rV;((ZG@Sc@h>2O`Pf+)xDApBUuWU)6SIF6;&WSrKTn8n=<p0978J1iQwkGH+@5%q76ljGU8yNd9NqOcEMtSD5FmmUbwrs*G4v2?il_kbLOC3bnLK6X`C!?l8?n3xwOm3r<ARJ#c5O@*AoGPGk3`}EqwYgLQ`vV%*4h%o)b?nqSsRgSZNxX*dVKy8Db~svOto6#^2sCtRg80whBhXNFoBZgBXU^l-DkWe<P>sM`WhVSeqAt7`i#1AUUHwxc-(JwjaG+|{O5i3p5KWJe5GJw+0=<4kgO0cZq(@OcN%V=Vvzi`9$<gRgf{*u4HTE-RkO_S2e?LEhMMKh)_1RThK7vJuJT`86H)e5*$juAl^;b-d6ir{ep+^}Pcoe;8g;Gb?art<U7^!FAuSSf=^up0X&b_s|p#L-qGr$IpJV<1zhCQ3*EF8gWpth<{_rJY`>$AWl2M-VG4MYz;rsI@4*6ySNjTNI$`_T)d+$IjM;Uaby(c_m~@AISEme)-MIYuq`%owBdU9`WTwin`@_S|O^uMdXvOaxYrMD3y(g*A96rU8(6bgT9Zb-FUS*iG2wDd!81vK38bGMZ-h<}RbUfkr~9$N)N;xds9^6}EWpnl;7*st5UPDlUdbb9582z6EX@pa)nIwSjXMMYXaP?o6Y;6tiR#1IwimN-qqgzI;KKZ+~&uuRZ?KJ9)p7VfIeVf0{@Z63SZ82pch-Kz@C2o4dBJp8I!kUbd$h5`y4o%U4z0C1WL$oRJkE+8{LJJN5P*q-y0r{zsy1M%`fihMlX2Ns?~z9sp1kHp+Y^yrdkB!`eTc>nLblkGf3~q_W0paPc}Fcju$2OQ{~Dt++mr#^_%&6e$OT#JJBPSxAPLrghGSm!>Ui{=WY4`@g?kOHU;crKO(N2gKd?WV8Ac#u(%#J)*ZILCW;!yPw(WCVPihO`A`icn!R<=9Jtig`=HgmpwE{EgAShx?aQfCemO@n6O<7X)5_gp`SA8EhTHzjI~cT?}d4z$*5j(<np<lPjVFmN-`p?=%I6SrBhXJ74-mX&TQ5SxEafx(9)Dn-odRqc)Z$77`r35yfEK0z`b^n-PX`gYsq*hH{rbuLa^-^u+O9Vq?&ZCV&N0>P;3>OlI}`eTdemPJG3;7s}oq9{BgY1*=Tn;^0KH<J~Iw$R;(SLhqulx7;HL|KwKw3p0N+hrHG1Jq+To{lI5i*QA*K9SoFN4w!>nPd~FXw%F@l|(GyR-B>C9bYnYRa+rS7I4kWwoQ_jc1^OzEuEo2c`lhm4>lK_HNl6vQpn!-NSG!o*|1}hk6n5^lB*ejDFC7uH8(b~L|p88d~rnaI*Yl~F9i$N&cVh{@LL2BsiS0ZB%%A8V2Qf;1aV2R5f@={0giM+Be9x4*2#P>H?j&g>hhrlQ<``KlvGcA>;Z^~r~)U16)YG`l@5FlEq`Y-~%*Ez9R2AM&;9{Ljo8~k;i5Pv25{XGv=$hTQbER%Y-tJ}9`^RpR!aTdJD<?_|r!is9{v|^m>&T_n4=>ace!*fsA{VojkDmt!LJ^VNHu2IQX<9l-YwfN5K`C1Mu#<M`3gbcNnPF2}W$8-3!gCotm;%Iz*>BACgb6w1z>`nyZ5SfRdz`9us(m{8agp%}T4;p(`i<^w%KSKS8r3Kgvsz4gzlN)zSoLB@i(#L?7a||XsM&4xpk&zA=hZd{@N{R0c9&~N@E8Dzg)Dy~nOj^!QQ&)Vh6&Sf8PiuIVI9pDth}?6$rQ+x_Cj!|Bc@)9JnA{Gw8hc)25KXhzxkFesK#Pv1X^!VMTr?fqV4)4lmsTrVx^veVjP1g>_~Grl3&TLt#jBsLIK{0yqFnHW-RhF&hDLcL)WT%qj;9j^HkEWa21ZQGhj5fKp49i^kLZyh9nnG*`o;C@%XbmdIKmH~(>fmIT)qa`w7`6L(CxHi@EoO#u7xYj`Hg)>AujtJmwl>D>mkMI=@!%X#~&Od?NYDTV;*RpSKv(@Gn9!rO;J_)!L`VGRT8)+>3j4<zyYa1)4G)`%^R`4EBGQg1Lu-+$=C|`j1?}_Q-*EK0BGpv<ESh8SrM6-H+9^zajo$rjkv(Y!3g})^Cao^oM8y9@^jTGJ2pD!eqv=d_iiyyLYMO-gcXIU5}qGb%1=;G%%k8eO`|hj#OIRFkbRDD+%4g;=j0A{Z8AG{*ttuCeQP#dsa*2=W?q`zlFFqIJ9g}{)G(66peAC1q!qazcaTMcZ9Phm(xP}Xs1>BoOO;_1U)+hHIsMP>^+<+9ThYds8Be6VCuva{eo)6a&?1J!c%Pw2Iv=&^3_w&$`s*x<)AT84RZ5WA`*DQyOYdoyeOMzv6A8Rlst5Xp^b&~DgTDB`%(*iYag$ZPFKM4?Djh(=AoEupeMn6f1Gv;NHKf<?|GVLG{%|c|;nho2bAx$Fc<awuRO1xQsw$&aC-b|H3~eFj2{T8EpoQa<6>HRs29f)H=ejC6$gUdE$AS2;$Px^b=F&jDWOqNJ9@Xd@`(>UI+^CgQuAZ@0X{|@J$B`8i9ID@7buApaw^TSWm`C9wP6$TCoPYUJco-S+>!O@M2bHHorP4!7riT^0nhrm(ASG1U(A1$#bt|X9H2vx}oDGgDk=Crvz~=6iew-FS*n&h5YE9dgEyCp=Q+2=A?q#EeV!pMenW*CuQ;1ck8{AOqpNGR{KC8oIti#sa4remMzxe$d|Nf1CFZnmlur8PHa``To?{fJrm+x}<E|>565ScE2?pI95hG93!lLYA7Y+>+Nv84}{b%=*vyA#IPZ$*eNukoK#7F~6moY1%`b>)y^)o4<J>{oz^-G;wdz9eZ!zvm5US38&hZZ~g(3(wKevHRFxtFlirWKYa3aNC)s^koU(j$hE1Wkp_i@>05_{h`{uOw)Ms_80FF=nB2Hy{FQpPOrDa-t~;<$7(#si%!M&&s}Tfdyn&P8zwKHM&_uyjf3HkCv1m$s=K?M>>uA&TK~sr2*$d`dt)Xfq{X-D-mQDN)$nfpCbtf~Tg!Z{NA_XY)cVZ3rL{iyZfUJAyjvyIM{vxoQ<pJAJ|@|BT~c#?ei}cTRK8#}9u-_Uf=HT%x<&J4YSW=M+Ou)Y!$tice5gkDcbFbKqVCEff4s5Fi-K|f@rJzZ@LqLGhu1OlXR>OJku}H;oVRT?OtT+xKn%N`Dl85&+f0B1yZ_J$xFpk>G<0BR_KBP742Q|EX;{3NUnlvZ);QXiLIwzwP5sL09T|S8R7fqzs9uw%Hsz$)`yejXYXV3IS*;klOLPLVA|FP=_|6z98H}J2w4Yi47fnE06tG(i1w^c+=I#>9pjU1r@gK!p%`v3nKk>J+BaVKjox3BCGmO%7p`x<Ql6Ak97uzl_M_eI`zn=7~!Y}BC2_<}@&pp8x|J`qT`6~z>*P<(zMu%ZVc1m{j;E#wP6K#F(D@ROD7Ee`BnmU)ym8JP0JxFx0w;UhTeeh>O2zi8+iXPmDcZHCGXY=n0Ed{sd-(8kr&F0A3q_&B!L4xK^0Ai)SvE*-3av7w;Q1UZ@p>7N5o0vSmlY{&1h0U>6UaV0{9;f~)GqjN%HY63Dm3u;O9d?Ata7<)@(u;4DiA%X0%!5qzUsbpHcisE{>UHh9`fG`F_QB^xf8DzBOE@ZTACyp+%C=n*@H&g#6ZsXXOh91PY`aBa%Pk7qZBf{2i+^vkaxNs?WyVhP2h<Qw&Zs}%J-7)X;%;)r|FaGA^)g6K{5aa4e_$gc9hK~05l#jL;pB+<==_e9+ssnux2Ap6_>kYD_R-;olFjNXINZM~Zg3LPFO)YXm7Uk*ac+$MnWDy*2^!v4KaToow(c8u5jbpSr@n!6pU|HcU)JTa{vl^RfyLmN^7rG3Km}wI>H~7n0b*k{$V53SW`QMugVfSzyx9`x_?ChTkNT5&cbKSmBZE-yJxB|kQIbwOy(F4^?(OaUCkI9MljtOy%z7W6lJt{|Q{wm!KAw6H&u)_O)84$_dHU)v-=Dpj(CDl(6xlOaK0F?%v&~M}><}Ky^y)MAK6&XN7^rNsxfX{ZZzWI!46ekDKi6RsJ@S=v3NJ19H&aehs(BUw?dGb(2*mQ91x+cbnxi&;yopXD#03T|sn$ea`&<1TyJ7=tz2!C3^%|1bt}jZPYIuPP{Pg0cePe5X*S@$h8j{>7(YKtUo$v5fP5Q<7cV^Dm-?5D;oA!s!-lLW#g|+ld^9mzqV87MF@!j`Vw0HplFdq+*DUsIzZ5*mJu2=T<7#zgEX;R-!#tdQY^<XkKp!D#Uac|D%V@cMqNP4tRWvw=wv?$Zb>@-@^s$c)q0%ipFqgDRKpr7Utc*5t|GXDnA0tV;fuNh9<Z^YQSE1cr86L`W51_~$qK3KvvE%p=1RhC3D4*H#}1Fu*pE2al&S*d!y)SX{Uj3!j95fqa~&r7O1EHqkH>ZsW0c}b(sij5BESy&$vVk-MQU*&i*8!cVg5YL5Pcv132XSZJEfhP~WaGY;pdA_Omw`KLuH@r}<6&u)CzMgM^^En@2j}y^6jp|50$jy#7RnsUoyRm$3igEHuK!R<GN7B{ABLz@-Ih^S9Ao|jQ8F)H?L#NGk%Qzp=XfEKF_mZ$#sP~&6-bUY6a^(Exr=MT_x&7+TuV4Kbes?wc#+XsZNuuUPT&Y8tv`mAAudhb@Q57*G`gIy|shb~rWjr4xL$s2kx^TYD9~G~~e$2o5c(+H!#SeeG$OCni^w`3-3WR8fuJo{GPBV7>(#~Vx(=VLsboaTo!iQ;RJRc6}k*mnua0jB~Z8nznDN9!6TXL1_nvF@eFTew&LU+mvE(`P{TNXCp8xt+KlGRe<_ZbqIZzD)sf}719{4RBD|M}{tv+wikg)up)Jz4=_H#r!nVx6q<Z~zq3aW=S{Pv*qr(2}(u%sZgV#`g1wPD!2<$wh2l_{I;-V9%IASNr_fuGd<T1ob#XV$f|FF9yCsT2OhO@^I@2dh0%{$Ny-rF>iA~2!ra{ta|JP`b1aQU*-vYSKqkXo0Xh$9*MPPeCf2dqOaTwmKT48*RDJiD7lwNT@oxi)1}-Ax1Q4EsX`;93T0%Wz<V*@`14P{Tvr@{*g_DDi?t8{`I^}E!BT+L84rOJ*hrDI{5h9jot1IA+{eNj5*%d~4*mbE-jGM)7*jTHCfFq>BbO6O^R(gf+1FP;-CT24pT_|u{iUKKTrBBREFBmATuY{c&k63@FDWBw6X8N~D~awF_FKOzxq*3_+!1vMX1@q2=vHGuc!`(w`lgIG`w;?`&4=$)sUT%1p_~|H;XUz^)v=N|_Ifs(5Xp)X=+<!Kf_V4qBs(*hZz5$`06ukBuG+G;;G53sdvWx-=E&bS?5286ovI2sO-IJ{#&5U2W4r9%{D+zqc+aF}fIbs-WYlp)12+2<wQ?<}x0(j7lQo;T+icz?*{k_%mW(q<vQjDzFsMm-$~aVvaSVvaEjlFdz^MQd5zFCl(*eID5S0BW28Cxnh)$(+1d|x8K>+(DQz*Z&=q9%u0jxf?+FsIGm}n=7{sj^y`~7NuT?f(Fp>p#v=TPQ^Kt^oxm%|l{ZX}-H4(^Zy;1(<Rcg@g6R|l$4M6UM1=rFeHnB5#Jt{f{Ry@Dd1-duS_)GZJK{5e)B_R7dTR$-BDX02TR?Bvt5s5%7Gh+~M+7E4OL^W;#Ovuj@p^{ZHPh6mHl#m@kQmq^@)?J(96XH&}DJvtoD^mg6(N`e_V6`b1hA1>ciybk}!F6ysW@gsV@8kKK?sgE(Z>$_&-PAg^k)=Ilo*80iGiFIq$HW;Nd9YKaZJtaEN4RGH|yRcggB)#wb4<cBpqW'}

CASES = (
    (1, "decode 1", "normal", 2826763372427500438),
    (8, "verify 1", "normal", 2827019192515550467),
    (16, "decode 16", "normal", 2826848645790183780),
    (32, "decode 32", "normal", 2826933919152867123),
    (128, "verify 16", "normal", 2827104465878233812),
    (256, "verify 32", "normal", 2827189739240917158),
)
VARIANTS = ('BASE', 'STRIDE')
WEIGHT_BYTES = 3 * 4096 * 4096 * 9 // 16  # codes 1/2 + scales 1/16
BUDGET_S = 235.0
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









CPP_EXTRA = r"""
std::vector<int64_t> okc_diag_info();
void okc_diag_stream(torch::Tensor buf, torch::Tensor out, int64_t grid);
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

void okc_diag_stream(torch::Tensor buf, torch::Tensor out, int64_t grid) {
  md_stream_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)buf.data_ptr(), (size_t)buf.numel() / 16, (unsigned*)out.data_ptr());
}
std::vector<int64_t> okc_diag_info() {
  cudaFuncAttributes a{}, b{};
  TORCH_CHECK(cudaFuncGetAttributes(&a, okc::moe_fused<false>) == cudaSuccess, "attributes small");
  TORCH_CHECK(cudaFuncGetAttributes(&b, okc::moe_fused<true>) == cudaSuccess, "attributes big");
  return {a.numRegs, b.numRegs, (int64_t)a.sharedSizeBytes, (int64_t)b.sharedSizeBytes,
          (int64_t)okc_smem_bytes(), (int64_t)a.localSizeBytes, (int64_t)b.localSizeBytes, okc::NTL};
}
"""





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
        _FS.append(torch.amax(flush))
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
        return med, low
    except Exception as exc:
        error(f"bs={bs} var={variant}", exc)


def variants_for(kind):
    return ("BASE", "STRIDE", "BASE_TL", "STRIDE_TL")





def timeline_source(src):
    def replace(old, new, count=1):
        nonlocal src
        assert src.count(old) == count, (old[:90], src.count(old), count)
        src = src.replace(old, new)

    def section(name, change):
        nonlocal src
        start = src.index(name)
        body = src.index('{', start)
        # These function bodies have balanced braces even in PTX strings and
        # preprocessor arms. Function end is the first standalone closing }.
        end = src.index('\n}', body) + 2
        old = src[start:end]
        new = change(old)
        assert new != old, name
        replace(old, new)

    anchor = '// ---------------------------------------------------------------- routing prologue (every CTA, identical result)'
    helpers = r'''
#if OKC_DIAG_TL
// [CTA,20,8] uint64: rows 0..17 are warps, row 19 is range metadata.
// cols 0..5: entry, W-ready observation, A-done, B-first, B-done, exit.
template<int COL>
__device__ __forceinline__ void md_stamp(const Params& p) {
  if ((threadIdx.x & 31) == 0) {
    if constexpr (COL == 0) {
      if (p.dbg) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
        p.dbg[((size_t)blockIdx.x * 20 + (threadIdx.x >> 5)) * 8] = t;
      }
    } else {
      // Keep pointer/address temporaries local to the stamp, not live across
      // the WGS phase. TL's grid_constant parameter also prevents Params copies.
      const uint32_t a = (uint32_t)__cvta_generic_to_shared(
          &reinterpret_cast<Smem*>(smem_raw)->md_trace);
      asm volatile("{ .reg .u64 p,t,a; .reg .u32 b,w; .reg .pred valid;\n"
          "ld.volatile.shared.u64 p, [%0];\n"
          "mov.u32 b, %ctaid.x; mov.u32 w, %tid.x; shr.u32 w,w,5;\n"
          "mad.lo.u32 b,b,20,w; mad.wide.u32 a,b,64,p; add.u64 a,a,%1;\n"
          "mov.u64 t, %globaltimer; setp.ne.u64 valid,p,0;\n"
          "@valid st.global.u64 [a],t; }" :: "r"(a), "n"(COL*8) : "memory");
    }
  }
}
#define MD_STAMP(col) md_stamp<col>(p)
#else
#define MD_STAMP(col) do {} while (0)
#endif
'''
    replace(anchor, helpers + '\n' + anchor)
    old = '  short pos[EMAX];   // entry -> pair position, -1 if not local'
    replace(old, old + '\n#if OKC_DIAG_TL\n  unsigned long long md_trace;\n#endif')
    old = '    S.xbuf_ptr = xbuf;'
    replace(old, old + '\n#if OKC_DIAG_TL\n    S.md_trace = (unsigned long long)p.dbg;\n#endif')

    def skinny(s):
        old = '  for (int i = 0; i < nU; ++i) {'
        assert s.count(old) == 1
        s = s.replace(old, '  if (PB && nU > 0) MD_STAMP(3);\n' + old)
        old = '    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));'
        assert s.count(old) == 1
        s = s.replace(old, old + '\n    if (!PB && i == 0) MD_STAMP(1);')
        old = '  seq = seq0 + nU;'
        assert s.count(old) == 1
        s = s.replace(old, old + '\n  if (nU > 0) MD_STAMP(PB ? 4 : 2);')
        return s
    section('__device__ void skinny_ring_ws(', skinny)

    old = '    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);'
    replace(old, '''#if OKC_DIAG_TL
    // One redundant first-slot wait, outside the register-heavy MMA phase.
    // No CTA/grid barrier; normal wgs_seg waits on the same full slot again.
    if (S.wgr[1] > S.wgr[0]) {
      wgs_wait(S.bars + BW_F, 0, -1);
      MD_STAMP(1);
    }
#endif
''' + old)

    def phase(s):
        old = '  WgsPend pd{0, 0, 0, 0};'
        assert s.count(old) == 1
        s = s.replace(old, old + '\n  if (PB && g1 > g0) MD_STAMP(3);')
        old = '  wgs_flush<PB>(p, S, pd);'
        assert s.count(old) == 1
        s = s.replace(old, old + '\n  if (g1 > g0) MD_STAMP(PB ? 4 : 2);')
        return s
    section('__device__ __forceinline__ void wgs_phase(', phase)

    def helper(s):
        old = '    const bool PB = q >= nA;'
        assert s.count(old) == 1
        s = s.replace(old, old + '\n    if (PB && q == nA) MD_STAMP(3);')
        old = '    } else {\n      const float gsc = p.gsB[ce] * TWO62;'
        assert s.count(old) == 1
        s = s.replace(old, '      if (q == nA - 1) MD_STAMP(2);\n' + old)
        old = '  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier'
        assert s.count(old) == 1
        s = s.replace(old, '  if (b1 > b0) MD_STAMP(4);\n' + old)
        return s
    section('__device__ void ws_helper(', helper)

    old = '  DBG(0);'
    replace(old, old + '\n  MD_STAMP(0);')
    old = '__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {'
    replace(old, '''#if OKC_DIAG_TL
// Address-taken kernel Params otherwise materialized a 272-byte local copy
// in NVRTC TL (verified in SASS). The annotation preserves the launch ABI.
__global__ void __launch_bounds__(NTL, 1) moe_fused(const __grid_constant__ Params p) {
#else
''' + old + '\n#endif')
    old = '  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }'
    replace(old, old.replace('return;', 'MD_STAMP(5); return;'))
    old = '    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    return;'
    replace(old, old.replace('    return;', '    MD_STAMP(5);\n    return;'))
    old = '  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered'
    replace(old, old + '\n  MD_STAMP(5);')
    return src


def compile_unit(ns, src, kind, variant):
    """One thin-host CUDA TU; exact incumbent math/flags, unique linkage."""
    defs = tuple(ns[kind + "_DEFINES"]) + ("-DOKC_PRMTSC=1", "-DOKC_DQ7=1")
    source=variant.removesuffix("_TL")
    if source != "BASE":
        raw=zlib.decompress(base64.b85decode(CUDA_B85[source]))
        assert hashlib.sha256(raw).hexdigest()==CUDA_SHA[source]
        src=raw.decode()
    if variant.endswith("_TL"):
        src=timeline_source(src)
        defs += ("-DOKC_DIAG_TL=1",)
    flags_as_source = "\n".join("#define " + d[2:].replace("=", " ", 1) for d in defs) + "\n"
    code = flags_as_source + src
    # torch/extension.h drags Python/pybind into every CUDA TU. Only this
    # equivalent Tensor alias is needed by the unchanged C++ launcher ABI.
    old = "#include <torch/extension.h>"
    assert code.count(old) == 1
    code = code.replace(old, "#include <ATen/ATen.h>\n#include <vector>\nnamespace torch { using Tensor = at::Tensor; }")
    info = CUDA_EXTRA[CUDA_EXTRA.index("std::vector<int64_t> okc_diag_info() {"):]
    code += info
    if variant == "BASE":
        code += CUDA_EXTRA[:CUDA_EXTRA.index("std::vector<int64_t> okc_diag_info() {")]
    stem = "md_" + kind + "_" + variant
    code = code.replace("namespace okc {", "namespace " + stem + " {").replace("okc::", stem + "::")
    renames = {"okc_run": stem + "_run", "okc_set_pdl": stem + "_set_pdl",
               "okc_diag_info": stem + "_info", "okc_diag_stream": "md_" + kind + "_stream"}
    for old, new in renames.items():
        code = code.replace(old + "(", new + "(")
    return code, renames


def build_kind(ns, src, kind):
    """One C++ binding per build, separate CUDA TUs compiled in parallel."""
    from pathlib import Path
    from torch.utils.cpp_extension import load, _get_build_directory
    units = [compile_unit(ns, src, kind, v) for v in variants_for(kind)]
    declarations = []
    bindings = []
    for variant, (code, rename) in zip(variants_for(kind), units):
        decl = ns["CPP_SRC"].replace("#include <torch/extension.h>", "")
        decl += "\nstd::vector<int64_t> okc_diag_info();\n"
        for old, new in rename.items():
            decl = decl.replace(old + "(", new + "(")
        declarations.append(decl)
        for short, old in (("run", "okc_run"), ("info", "okc_diag_info")):
            bindings.append(f'  m.def("okc_{short}_{variant}", &{rename[old]});')
    stream = "md_" + kind + "_stream"
    declarations.append(f"void {stream}(torch::Tensor buf, torch::Tensor out, int64_t grid);")
    bindings.append(f'  m.def("okc_diag_stream", &{stream});')
    cpp = ('#include <torch/extension.h>\n' + '\n'.join(declarations)
           + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n' + '\n'.join(bindings) + '\n}\n')
    tag = hashlib.sha256((cpp + ''.join(c for c, _ in units)).encode()).hexdigest()[:16]
    name = "moediag_v10_" + kind.lower() + "_" + tag
    directory = Path(_get_build_directory(name, verbose=False))
    cppfile = directory / "binding.cpp"
    cppfile.write_text(cpp)
    files = [str(cppfile)]
    for variant, (code, _) in zip(variants_for(kind), units):
        path = directory / (variant.lower() + ".cu")
        path.write_text(code)
        files.append(str(path))
    return load(name=name, sources=files, build_directory=str(directory),
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   "-DOKC_NS=3", "-DOKC_NSF=3"], verbose=False)


def jobs_from_counts(counts, kind):
    """Same job order as route(): fat (>8), then skinny; MID splits all by 8."""
    if kind == "MID":
        return [(e, min(8, c - q)) for e, c in enumerate(counts) for q in range(0, c, 8)]
    fat = [(e, min(64, c - q)) for e, c in enumerate(counts) if c > 8 for q in range(0, c, 64)]
    return fat + [(e, c) for e, c in enumerate(counts) if 0 < c <= 8]











CAP = 64  # CUDA_SRC: OKC_FT_T=64 -> FT_T (max tokens of a fat job); MAIN has OKC_SPLIT=0, SK_T=8
NREPS = 40


def local_counts(ids):
    counts = [0] * 32
    for row in ids:
        for v in row:
            if 0 <= v < 32:
                counts[v] += 1
    return counts


def _add_tokens(t, n, pool, cap):
    """Add n tokens to pool experts with t < cap (fat >8 first, lowest count first)."""
    while n > 0:
        cand = [e for e in pool if t[e] < cap]
        if not cand:
            break
        fat = [e for e in cand if t[e] > 8]
        e = min(fat or cand, key=lambda x: (t[x], x))
        t[e] += 1
        n -= 1
    return n


def retarget(counts, mode):
    """New per-expert token counts: same touched set, same total routes.

    NODUP: every expert <= CAP tokens (one job each).
    DUP3: exactly three experts (the three largest) with CAP+1..2*CAP tokens
    (two jobs each), everything else <= CAP.
    """
    total = sum(counts)
    touched = [e for e, c in enumerate(counts) if c > 0]
    t = list(counts)
    if mode == "NODUP":
        over = [e for e in touched if t[e] > CAP]
        excess = sum(t[e] - CAP for e in over)
        for e in over:
            t[e] = CAP
        pool = [e for e in touched if e not in over]
        assert _add_tokens(t, excess, pool, CAP) == 0, "no room to redistribute"
    elif mode == "DUP3":
        top = sorted(touched, key=lambda e: (-counts[e], e))[:3]
        assert len(top) == 3
        for e in touched:
            t[e] = min(max(counts[e], CAP + 1), 2 * CAP) if e in top else min(counts[e], CAP)
        pool = [e for e in touched if e not in top]
        diff = total - sum(t)
        if diff > 0:
            diff = _add_tokens(t, diff, pool, CAP)
            diff = _add_tokens(t, diff, top, 2 * CAP)
        while diff < 0:
            e = max(pool, key=lambda x: (t[x], -x))
            assert t[e] > 1, "cannot shrink"
            t[e] -= 1
            diff += 1
        assert diff == 0, "could not match route total"
    else:
        raise ValueError(mode)
    assert sum(t) == total and [e for e, c in enumerate(t) if c > 0] == touched
    return t


def assign_ids(ids, new_counts):
    """Rebuild local slots so expert e gets exactly new_counts[e] tokens.

    Per-token local-slot count and non-local slots are unchanged; locals stay
    distinct within a token (bipartite greedy: largest remaining demand first).
    """
    rem = list(new_counts)
    out = [list(r) for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (-sum(0 <= v < 32 for v in ids[i]), i))
    for i in order:
        slots = [k for k, v in enumerate(ids[i]) if 0 <= v < 32]
        pick = sorted(range(32), key=lambda e: (-rem[e], e))[:len(slots)]
        for k, e in zip(slots, pick):
            assert rem[e] > 0, "routing not realizable"
            out[i][k] = e
            rem[e] -= 1
    assert not any(rem), "routing not realizable"
    return out


def verify_routing(ids_old, ids_new, expected, mode, base_counts):
    assert len(ids_old) == len(ids_new)
    for ro, rn in zip(ids_old, ids_new):
        assert len(rn) == 8 and len(set(rn)) == 8, "duplicate expert id in a token"
        for vo, vn in zip(ro, rn):
            assert 0 <= vn < 256
            assert (0 <= vo < 32) == (0 <= vn < 32), "local/non-local slot changed"
            if not 0 <= vo < 32:
                assert vo == vn
    counts = local_counts(ids_new)
    assert counts == expected
    jobs = jobs_from_counts(counts, "MAIN")
    touched = sum(c > 0 for c in counts)
    assert sum(c > 0 for c in base_counts) == touched and sum(counts) == sum(base_counts)
    if mode == "NODUP":
        assert max(counts) <= CAP and len(jobs) == touched
    elif mode == "DUP3":
        assert sum(CAP < c <= 2 * CAP for c in counts) == 3 and max(counts) <= 2 * CAP
        assert len(jobs) == touched + 3
    return counts, jobs


def routing_cases(bs, data, modes):
    """[(name, data, counts)]: the generated routing plus synthetic variants."""
    ids0 = data["topk_ids"].cpu().tolist()
    wsum = data["topk_weights"].sum(-1)
    assert bool(((wsum - 1).abs() < 1e-5).all()), "topk_weights not normalized"
    c0 = local_counts(ids0)
    cases = [(f"R{bs}", data, c0)]
    for mode in modes:
        t = retarget(c0, mode)
        ids1 = assign_ids(ids0, t)
        verify_routing(ids0, ids1, t, mode, c0)
        d = dict(data)  # topk_weights unchanged: already normalized, slot-wise
        d["topk_ids"] = torch.tensor(ids1, dtype=torch.int32, device=data["topk_ids"].device)
        cases.append((f"R{bs}_{mode}", d, t))
    return cases





def run_all():
    dev=torch.device('cuda')
    if torch.cuda.get_device_capability(dev)!=(9,0):raise RuntimeError('Hopper required')
    ns=incumbent_namespace()
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    log(f'MD_INFO v=13 READ_FLUSH(judge-like: amax over 256MiB, no dirty lines) STRIDE_IS_TINY_L2H1 device={torch.cuda.get_device_name(dev)!r} reps={NREPS} flush_MiB=256 BASE={INCUMBENT_SHA256[:12]} STRIDE={SOURCE_SHA["STRIDE"][:12]}')
    log('MD_NOTE skinny A_then_B strided; per-job A credits; Bfirst=phase_attempt_before_readiness_wait; TL=separate_TUs; pass2=reverse_cases_and_variants')
    ready=queue.Queue()
    def worker():
        start=time.monotonic()
        try:
            bundles={k:build_kind(ns,ns['CUDA_SRC'],k) for k in ('TINY','TINY2','MID')}
            ready.put((bundles,None,time.monotonic()-start))
        except Exception as e:ready.put((None,e,time.monotonic()-start))
    threading.Thread(target=worker,daemon=True).start()
    raw=generate_weights(dev)
    ns['_ext']=lambda defines=None:None
    # Avoid the board selector: compare pinned TINY/TINY2/MID builds only.
    state=ns.get('_moe_sel_original_prepare',ns['prepare'])(raw);del raw
    ns.get('_MOE_SEL',{}).clear()
    assert state['sfx']==('-DOKC_PRMTSC=1',)
    flush=torch.zeros(256<<20,dtype=torch.uint8,device=dev)
    entries=[]
    for bs,load,dist,seed in CASES:
        if bs not in (1,8,16):continue
        kind={1:'TINY',8:'TINY2',16:'MID'}[bs]
        data=generate_input(bs,dist,seed,dev)
        counts=local_counts(data['topk_ids'].cpu().tolist());jobs=jobs_from_counts(counts,kind)
        entries.append((f'R{bs}',kind,data))
        log(f'MD_CASE case=R{bs} build={kind} jobs={len(jobs)} routes={sum(counts)} '+
            'cnt='+','.join(f'{e}:{n}' for e,n in enumerate(counts) if n))
    log(f'MD_SETUP seconds={time.monotonic()-T0:.1f}')
    try:bundles,exc,secs=ready.get(timeout=max(.001,BUDGET_S-(time.monotonic()-T0)))
    except queue.Empty:log('MD_ERR build budget_not_ready');return
    if exc is not None:error('build',exc);return
    log(f'MD_BUILD kinds=3 TUs=12 seconds={secs:.1f}')
    calls={}
    for kind,bundle in bundles.items():
        for v in ('BASE','STRIDE','BASE_TL','STRIDE_TL'):
            info=getattr(bundle,'okc_info_'+v)()
            log(f'MD_RES build={kind} var={v} regs={info[0]}/{info[1]} smem={info[2]}/{info[3]}+{info[4]} local_B={info[5]}/{info[6]}')
            ext=types.SimpleNamespace(okc_run=getattr(bundle,'okc_run_'+v))
            callns=dict(ns);expected=tuple(ns[kind+'_DEFINES'])+state['sfx']
            def selected(defines=None,ext=ext,expected=expected):
                assert tuple(defines)==expected,(defines,expected)
                return ext
            callns['_ext']=selected
            calls[kind,v]=types.FunctionType(ns['kernel_fn'].__code__,callns)
    refs={};times={}
    for rep in (1,2):
        for name,kind,data in (entries if rep==1 else entries[::-1]):
            for v in (('BASE','STRIDE') if rep==1 else ('STRIDE','BASE')):
                if over():log(f'MD_ERR case={name} var={v} pass={rep} budget_skip');continue
                try:
                    for tensor in state['scratch']:tensor.zero_()
                    outputs=[None]
                    def fn():
                        outputs[0]=calls[kind,v](data,state)
                        return outputs[0]
                    if rep==1 and v=='BASE':refs[name]=fn().clone()
                    med,low=bench(fn,flush,n=NREPS)
                    out,ref=outputs[0],refs[name]
                    a,b=out.float(),ref.float();delta=(a-b).abs()
                    eq=torch.equal(out.contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8))
                    finite=bool(torch.isfinite(a).all() & torch.isfinite(b).all())
                    ab=float(delta.max().item());rel=float((delta/b.abs().clamp_min(1e-12)).max().item())
                    l2=float((delta.norm()/b.norm().clamp_min(1e-12)).item())
                    log(f'MD case={name} build={kind} var={v} pass={rep} med_us={med:.3f} min_us={low:.3f} eq={int(eq)} finite={int(finite)} max_abs={ab:.3g} max_rel={rel:.3g} rel_L2={l2:.3g}')
                    times[name,v,rep]=med
                except Exception as e:error(f'case={name} var={v} pass={rep}',e)
    for name,kind,data in entries:
        if all((name,v,p) in times for v in ('BASE','STRIDE') for p in (1,2)):
            ratios=[times[name,'STRIDE',p]/times[name,'BASE',p] for p in (1,2)]
            log(f'MD_SUM case={name} STRIDE_BASE={ratios[0]:.4f}/{ratios[1]:.4f}')
        for v in ('BASE','STRIDE'):
            if over():log(f'MD_ERR TL case={name} var={v} budget_skip');continue
            try:
                trace=torch.zeros((state['grid'],20,8),dtype=torch.int64,device=dev)
                ts=dict(state,scratch=list(state['scratch'])+[trace])
                for tensor in ts['scratch']:tensor.zero_()
                _FS.append(torch.amax(flush))
                out=calls[kind,v+'_TL'](data,ts)
                torch.cuda.synchronize()
                a,b=out.float(),refs[name].float()
                eq=torch.equal(out.contiguous().view(torch.uint8),refs[name].contiguous().view(torch.uint8))
                l2=float(((a-b).norm()/b.norm().clamp_min(1e-12)).item())
                z=trace.cpu().tolist();nw=8
                origins=[w[0] for cta in z for w in cta[:nw+1] if w[0]]
                if not origins:raise RuntimeError('empty TL trace')
                t0=min(origins)
                def metric(col):
                    vals=[]
                    for cta in z:
                        ts=[w[col] for w in cta[:nw+1] if w[col]]
                        if ts:vals.append(((min(ts) if col in (0,1,3) else max(ts))-t0)/1000)
                    return '/'.join(f'{x:.3f}' for x in (min(vals),statistics.median(vals),max(vals))) if vals else 'NA'
                log(f'TL case={name} build={kind} var={v} us=min/med/max_from_first_entry entry={metric(0)} Wready={metric(1)} Adone={metric(2)} Bfirst={metric(3)} Bdone={metric(4)} exit={metric(5)} eq={int(eq)} rel_L2={l2:.3g}')
            except Exception as e:error(f'TL case={name} var={v}',e)






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
