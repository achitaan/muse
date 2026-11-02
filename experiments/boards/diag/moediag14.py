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

SOURCE_SHA = {'STRIDE': '34c8d4c659179268b15c538b82b03517534b850fe93f5550f006f9deee8a1d23'}
CUDA_SHA = {'STRIDE': '34c8d4c659179268b15c538b82b03517534b850fe93f5550f006f9deee8a1d23'}
CUDA_B85 = {'STRIDE': 'c-ri}dw0`F)-U`&pQ46Y&=PTsWji6ngnknsz{90kBpiBHPR7R<*>U2?j%CLVG|y*$_iyhisY;S_X`rWjnpxAtlImW&cJ2HA=b)#$VQ&zHUGJAyPnvu0UccP;)!n-)dAjw_Swuzn#mBu{|CHkTQB~Z}JDKl0xdN}VC0^%?yvmh%r2@Oq-m8752tBJ{{nkIjs5|K0x^@4)TK&rZsmZ9-J#CHA)|1iccrXlCr-Rc_okXK{ti0)w^n6d9w}!JYR(NqVil?g8na*0n;e{Hv;y8>;YCN39XokkynhmG1>WwB`d=kb}U#&h??^=`AX{_$6z0>d%FXB^bK^=v|aX6vQy5ZTZ6-`TM<@B^wQm5_KWHJaRCH1<hjt0>*=5{8d*);U{XM8e<q6;+{M130Ud_=>X45*EGlZSE<wN3|})$wRFRC}+U{_8Du9#31-L8qkRFjVbuIGS@qz1EbzOv2Ww@;c*kE53+2YDJHvjHi;C_vyFkfol<D$mrF)H5v1p=X=U~zqiU|c_!n@s5|R~llz@f6wgk>iOgXZHOEJ-IK;#0C~Wp-aoEMfla&=UY|Wz15j8tH=`^Wm84aCRH)(NOUG-0b=Juq85-~MDBdAUmdlb=@v}c21*V?O&>@07O_h$PgmSO+p+kfdlRr>qp<<pW1Tb&~{A54!_E9&wOT4}ZV6cF(4{hR&0CtE?f(P*9<>(MT@Tb+}J+W9EeqY>5eX^l^t4=XJmR71wzqlrxiv=Su^D+JpycSm(&s!Sv3(6Yn?Hm9TWuv=F9t&=cX9k;06&ZrxzL9DPkt*B35d537}npnemy%ntUI*sVW0H_A>G>oR|B%DOyaFZbiph9~x9Q9{mNj-n|=1s7&Hy=EI{Xw3pE753j%3wlJP2+0~72e#Z%`4yf=OF3~XEcgOomsclZ1;lo^3h}K_4HyKHm8%;U>ZNpmZq)3;c_d)(=I_azq5||(`P#$o&#lA?>_83Ywo|<296l?5+w0<`^gih7+$EV!4meD_pg81eyRecXwmdiY$_bV>1ce?91zG12_3|=!tsn?svQzkg}l!(PHa4-$M}{i`X9B39(D;vstvBf0Hz`FG;GDQ3E;a@Ub{nZF;b%*0r7>w7QQpi-Qd~w``7={Bc)*)kls|g+k5+goHiaM4VF+1#;?b&2R&?2Ta};aSx|@?L{!j4xX<@?U%uVj)7{YHCJ$V;9B2)viVzkb3r0HB9zg+(UB!f|XfXss@<CqRQ6Zf&fR&uW`D8Gq5*NO+UDV~1U)<Y4USV7FTlVq!8#-YRop-;!d;bij^($VI=Jxw1FEWt9AKadJ^=jFUq=jGJDlaYcSXIpZ$U8~Dt&a2;4Uz5)pS;`Mb$Z1MYj*YspZv17zvJGf`knQ=XkGZro>j^{J%s7qwV*ezvPE90dQsE+sGXiXxi>u1EX~~eN3trv{gzbmFI}R9)}SkVYfEk5`5&^GyUFjjzu<so&cc3kU#-`j$!LPbrVQ|gb_{?lcj9V#69`TY7laV$tPt=%rb$t&(-9$7g6Q#ZFs1V`9jQlK{AHEDsADDp_-6Q}GgZ}fzi1D!4(|G4VJbyM-+puv)kn2aRE!V*Tc@bJDn24RFQ32R6>^TqE{5PjWfmxlo7>&q-`n5bfB8fOd5BZ>N&x^Dz%LifPpyEm%-}e=yTS!sAr5V!u^kgsgtrM30jfHq@j&07bh!|!Iu9oohAJ}Xs2H#8{i^%!hUc4v1GkuVF_%925gJBS^(V%1F&ZC_!>&86=X=@Ka}%R^DMJ`-s=6}d5Kpg9=yhyB@fBcny}mjb&FQ8eOf_+-T0CV#PjS8M)b?J!{?L5-^8a`=?zXR*&0cqKKA?}D*CKkvYx^#)`tWi4{X6%P#T$Jk$ih?M1+52-uup`MijQW!-Z11#Y&>1GVxm_fqA<cSmtXCMlL65!hF~5<m;){nqG<Y{M<%fa83m)jL-Xa+U%e?N5Fho1P0*iSWzp0Bo6$_@<KxF_t)wbAmBTnJtnZm}g-2ItI%oRws`&WQJ(W2`{02Ln;+^ECmTQ!*z|)`Pf^N=<&K!?oo32O*e7bB`)J1#NGuL>FC_!39PcRSdCQ({`Atdj;eNW(o5sX`t>7X@S#T`ve^}toM!|6OEWQN{A2h%Oi`y_{kN#yjDx#)?pF)hU{;v&L$-0Fm&$JOUsS=fK}{&g1o=~eNS_w>WNcbWTNUa86=sJQ=|4TnWdyso}*>eI8+S5k<UgczssXgKHqOL$K{?7!Y7g69blG~o-v_<N@hEB>aMhlBnRT@9K6GO#|1XvSW6KA<h=4JO3JHE>PPT4CvUvxo*FuwI>jv>&#nOlKM1Af{ELvq9vgZe!S@CdvvO$4nv#y>;5s3<p0D)lV3VuwSn?===mWAZBh7Jy~SGnLkalx%Xjr_vNqZcKLO6bF+iFHNh<2b_)IS?%Dh8w@=mWlDboOKW)E%u1f0m@HQQ>pt_sg0{>RC*~JXPW)m1?H0gwcXh^l``Fu3!(z-Qq{+S!_n$75(ei0Z?gY|og5Rvmno%;kqO10uss!VG}jG1>ko|gN=QM)zF%%R+-8V7eOhcs$pz3x;S|2Mj=Ha9U?Ahg>@x4nClkauM~38zisou-fICK&X=6~;%S$@Jc1@9w$pCo`<)XQ(#SZCUx-e&g1cYfrdWt9m>yo+tmh>HKQdOXt_S?VaCf=q@g_!D9(L<Y3|3Z@0SFqC51wp}u})t)`3^2p8#;s5_j!se190zSk=c31z-CKO31cM0jm!3aq5w5<!YL_f)N#E4cp}ANL72d*)vMnXYD9bWOheOtt8#LZ|^4c*UVvypMan`Yk~(IO7ZV{wAbEZTehS3q6Peq7DtJCR`NaKJeU@e-nztZi@O&_^*Vd@Us(ozoVzlV^8jnO+{QQkD@af#qlaPSa6*r=#{E#+8?wJq-Xov39mUKDvYsN!9<uW_cHg6!UZ1-olc)p6l^45vUsNbYRV!|x6QqOy?x^OnG;84S!|_ntSaVMMYp}T`_7q3$%eh=|NQdu?ed1IxrSb6T5^t_HL@lXFnR^N<QC6t4e?A{fUWxkjSVAIATsck>Kx6YlUSv+a0EI`$PEyP$^(>hMIs(eF+V3e-N<ywT+o*JtO4M(wJ@9!k)cUH#0)8K1B7z4iiK&YV4a{wKgH^N&=M)hA`5fxpFM4UOuHfOYlA><gO;w|xFyoaCS<6SsW6r)Xc=aa@Spxb<%njco0?bv7S#?zYk)COpxZgKk6Q^Rxnz!ze)0r|Bf4HE!ujFGwRl6>gxutIn%mTVmCy&FoBOmVCA`Badm+!d%V3@9`Sx(sIf*lke2_-CEotI|G{G%#BL&?yU;px)$~=?~_0vXnNxeWldX$v#v7UIKCFRCXXYrJ1L{(i|&oAE2?(1zd%1zWNg5qZJa5UYdMKe+#hCsS}d!dl5E#aOB#<92}<lVN!(Q4Wn42ARoMWk3d&3uk*2G{s0Hv2jsO-^_(I>XlBRN5zAPqCQZOGFufdOC8K5-XS!4Wfx2Jv>BvC4TuFHw({Q%Fvvj(~g?P8$^XoXOl?FLVi|WS3RQ2^s7r?9PI%QRB$IPS^I5!?YCv^x0AJB3#wRqw=Q2#d%F-xJTStOw|jUbp69l~gxCu}y}%f4q6K8i16cFIW})`Yo9$m|j<t<?Nj;O_{OTUg+W+W}Rk~65<9#_rD*!IHomYE+{k8RmEeEAuH#HhsK#bg{ZK^LhKEJ-BV@YjLLufu}fNEkQW0r?|!uzB(g&f%b_{%zY^Llwb2-Y{&E5Sy+`k=P45e6H*#w}XU$&9FOt#tuSCObT%j_dc-e7oVkh?5tZ@aWT<T&<mKt({!09k*7k+N9UR+0+!wkW3J+&*f|pcQ|e0nRrEFtoy`BM|%2%Z)u6<kp6?2rgK2I!0tQRrK4~XrX3p6mpFw9CfRE&`Lfe%nAKh#O}e-=m`*V=6|c6gUhVKUo~-gV`pbKnE$wBdu?B(RDixUy`m@n2F6hKYkeBonE$u0io}v~VRiau>7>-OoqPv0;<&w7JrESLzBO})>ZmI)w+Ay`l174xSgM{)rJa|YmJEYDs+r1z#%f8xaq$2pS+&?A$A;)JyM^G3~+H;x74&lxV2&zoS(Z=Dy3H`_ar(UT%H4KX5qM?@mFV`#0i1F}Zn+5pOFfxsGM^T6+x?W?R%t7A~0zc+?J0(yPnl<M3iNv52&7;OXnd75Ag_Du2QENDFUBs%QywSLIMzk(n%|^CC)W9U&-s~bbQVf4D(fDKzugGZ{GG+Cqq=uuCiU=edW)(WqNyC~iEjFQkyd|_A;O`D3hKXFKrozQw>MUU+tqfFK03y8CGTj4UV6ZYMrUwbhLqZ_x0x#?Wak5LMz#+i)f(A-E%Nm<-qJ`W<PsNTxhYJRq)JQ-`r={5vy7t3Z8tGH5yDG4|3mTd3@^d0s#Tq6j3zQcBL#xv66X%*@Y*vqdwzr><Cyv`;gZYnt7`8)?fZ(NjK%C9t!P}3A2es<q0h64E8mH1!nMBERHavMvw%ct5hx-3XJ9=`=w`jTqxCHWO+_6OAPzNAVqLEN1bWnjTL>TC2g;O8*^z7E_G`o2m5*U@}@1$pb;EhHKbf8$jkGr0RHw;_Ql(#QfbLooo5)oi5XNX=1<EDC(Fw{^@tsm468<!tr)&<gCLnIZEVw#9!CQN>7vTp)$R8F2Ndy~wM4nurUtsh!Qu2L@7627=M8xBi!jh7!Jufo&u^rECT%H?2<(4UkD*30E;jUM@{nN1%=WICZ#lS!jb(G1pU7*0VxYN8(FH*Ivrd=%+|4eU`GY}J7g0Vv$T!XV1D=8j#}t+{ha&0|8eMKxK_NZZ)$(pYi5(Ai`KEcg?Gss&ht#g6hAQR{Mq12;_j4&IzkN7Ky0m`Tx>){qC7t+3N5sIX&KFh|DS;-IP*k9N22nx<&Vm2QAF+R}ipX7fd8T)!BIWQ$#Zb+-6M04nXp;aN%b_+Q!@QTAIucMSUPb)qTINyFCgv!>DL;elI3uSo39baFwfGlOtfwBou!8Xop;^|4k#68AaACPCgs*AgAj=9n%O-6{H9uM-j&^?LZxNFdE%H}mHBncN>Gb-b7SdSZU{n5L5_QzJUwrQ7Uy&pZ<*Z#c&u!|qQ8iKK;wPG!PL3nnXIy!2&2*E!9@Tj3SZFMWcP0oV@S7I_CQPHqkRwiq|;9*YZOc+6=E9!zLRMT29)$bv3LjNU5USrOf@4b8Uxrp`B)@~Cu>G|2e6GvW(7dbQ-*kk6Yodujj<g11gcd|J0;P95EwbaM=*av1rZFJ^?%pG91%aL|=26Pv>FYJie{r4>u`l8OVun3m;>``*S1=I!Gm5V%U}6Bpss7tkd58@C1%d>vprOKLSxbk=ZV5rNYh@)Zn*U^p60ycRKo9YQt4wi=yiD`Lt70us?GDh~%h!vxzVu@bg78J!ZE?Tm(4Xt~q@KP>j9+YJm0uZ*;UKI2(i404#rAO!fo0d2nzF$0G2NLBw`C0=#%T>URmSv0B=8i+8YaM2=~VTC3`NOpBLzDfXaIvY~I%{~hNTEhQ#v>3ohnkUbFi($m4F_EP_7ISva=mj;8;#GQXPCj`eu=dAPj2hsAJ>{Lz<8wue@2l0-GrR@z<WKaCkiX9b2nSM$l@%_b#SLdy=pquV-KI%Zz2uQ*m41?$IMi5GO?dQ2TrICZaSKC#0$s#)%C&vp8h?Ri#_+>bcEb7!ko0t=d+jnvZT#yq7S}J226%Zi@I_$MhP@_QBK!u!DwfRQ!;3c`URTwK;1QL*@{3CIMTG#R{32++2>k1>;_!PfW3xFN&8x|4@0!bc({;V}gpa=egpY`*`~EY&Qny5Wv*@h!P1Lcb+p5?Pf&EamA8PgzRZ6$ib^n*v3JMg2x}Dp`*tXmn@*Pn=4c4QL6QX>H+$$4CE%$2GGFx5pKlG$N->H=D1f@IG(w*9u5<S*Sch*XG)=PID@Z(15&O`oT6@wZ>_qnNVuk>zvorB6D6pivbuzpvq-?c+Otv0#swJP|AAA$K%H9u<lN86O4nO{r0+8H<5{+M=3Xe8(rN*Wo<n?Ap3<7z&qrRw@FzZeskCSdJ1o1OD%tI3qipwpa=nlatHVb?>uzCq~mCny%tnEIJm4VLKU^ti?}y4>lPr4g;y)k@rhJHcUzCHSc-mu4Po6)uLY7bE_3?ZqJ8<#aIhAfsM38ZOafR!4X1LmWLNP6=iS>Je+09$%-odF%4N@LhV!sKb2!3I%VT_g8&L@h6_%zWn!TehW@V=jGXYojRvQmc{2r8n*$HTeQkk2X#zYE3P$RPj{YsCqB_<G(X(Hsj2bZyIMv0vb!iRAgX8k&vCii?)&oRfMt{N{T?w~D~j4WB&ad7p!pMZ1D%vt@o$9-LRb--Fac0_LHJlVn_jT)yFx~W@#ds8KOjQzk~K_-=w@feJuW4p;ZMLGAF9=Z=#Zr<vpAf@SL#TH-DazEHXBSr&k~u9WtULkJijf&ZdunU_s271?Ev!zY!uFgzzS{A1U3!$MSC@N_UfGRIL+u86Lq`q>WbjE6Q06|z13}vK?gH=`8o{MyD&OAZ4n_m48x!j+^5?=5=(cnuJ5d^uZ6V-wQjAx_F%2E)_c(HZLC+@m9^eRJAAMnJghv}Xjcdfw#Oh%SF7cU&ldIjEKr<G!gxF)#*r=N!!hlmIES>sOoWru*;E*#0c`WygX2)>TxhX|Lnu(FU|hcP&~?x1=^cEQUNCpnq~bgU&QsNSs#z2(D@l?T-w=dW78}8n3jUk%a#!fxnYydq@KYeP>@3349SwTZ=Fp>Y(Eo#qpOmUDFI7uR)qr|Ev|U+LD!~r^n}NiIdDljLrc2T`e<S}L*P{WpRxkl2E{$mVF2T)!SQ~9c<p4e}eb|}|FPhAzYfu+;5gq&zpP0L9&u`?ai3iTo3K`yEI)Fma!Y(B-&)?!NcU6t{mBo@<mhB?lX$_~9XY_eoegn+oeqg%+&wEDrNSm>!Rd5yem8YM5Ut71>)dqk8?)DxvL5l?_I2}aZvl70o(948}!8jk8)_t{Co_7xL4wjXN2WA#Pg_TAkz!7f?t-6FTG?k#T-;?%@7><Y=3RbIk#QbA0WlEnI)n|OJh+zXiSY2BO<?&hT(xLxgJ4p}AvOy4~^~u_Ai2e~#M5q|$0)hBqS`0^_`H1;;*^&~U)&if8)ZN2XTE2b<m8GY3qjjwNzFG3O`wgxE<)8}B52);6BW*)=vQE{XVex27{yCns`ll_r4G4mVToS%a%$=U_g3_Ny=(9oG;fcVO+uoqAL1~4G-b;p{(EkZGg$g`EuRnwN==QxFF`cwkCvtjRHp&s=N2xAQ^=a!QY~s5|7)k%%mu|j%*C_z{^zhSqcFpsHI;QxU=SmmzDjgDVlJb0!dH$4np6ll?jkIaHjqAea>lYvglxJPmzUQ42xvU;*B7q+sQA5xa`LxXZVpFXKzTM14Q8R(vOpt9RC}`$WQ8QJ$nQFF~YC$veqGoD#Gqr3pwOliKzKoX-Oxa!;qAl9Hd#ZluQ)kJNez|4-P0SS?2vQrjEC?D1WHjgpK>_}(z$DK=1B!%vGXb(w*ahhNdUvH`J)pz<yu)PDp)=z!o$i-Zgq#|7HSBS=A>O02M$pbb9}z~?KUcOGZGO(+0nALk^L>y-6KKsFa61?bI`NY^U|@EIgqR$Mu$IGmoWg{>*@`X-%;!VBhd5s8?B!PCTfP^2+>$SdjWIqhi^GHP5c6p^$D?samyrd^t_<3iYmJ(>w~Hm&37UjnL0=q~;E}oDkvU&OKqGTD**Tc=cJjdOos;n%9^k{F80VlH8aQ4zGzju0M4cfZY&5-m0Jy?wKXDJzO^Uydwy-jv1q<<tv#-KzT(I8grKGwrAn)nn(xMaF%H6MMi&irx_y%str&Sr5z%O(IGNUc~H8ch0HT_Okc2qTQZoahc3Q{vPX=*RCf^-?fR&J9OIv$Nr`&N~V%_7mlFdp(nucf87T6phOs7ut*ZdbHAT0^4kQHUNGM6CTvhzfE2Lf3h7DL7tfQ$ekzv0cGDxZ<qF(~Du~3+-{+&mOs`cPyZoc^%`m#}Fh&llr#DZ(}zb4u)LfzS{7`fRN_?2*in48a5Kmqn;#j6XI|nKAc+hndsK?<3j7&b?s*}8n(>Rqv7A1mkwWhHR7jPbVxg-R)XQQ`V07pdM>G$f5peY^MshI9Oops;-K^fI3^`^UcI7V%mr;S!8%o+F?gIR8c!_0stxPjao2XA5ztasBMhkq@m3UHQ8Jd2=>g<zqoJ(obv))3-RGT)`85{RS;aFsjb|`0>>cAco~28;S3Z7rK#i`bGn`4jHSluJdI{t~Kk3>PgID_e#QF>hA0J8#1glBGl$y((w8F8)$=GC6anY#af>Ftt;ZeoTs044MqtYnhi*+Kf&D68iX|6+q1WgYHSK2qY1uyz;qSDdvuGyUiKkhNm?@L`&5;JaGX}I8QOO%wgNsK<!Q7c9q03gP*SvU**Td94dyb(0FN#mdNLxM&zr4}8swgd2pXhs}?X{$|N+85dipc^g{7=uSyg0oV(I|Q}K7~wt)EAjl8&*$nY{Z2tDsnM&*?)EASdOUCX?a(gn$-yOs5*GE67GT-zDN2~;*mc+wa3mT+Jz&qg8k@z^kg^RV6b+=SICS)PIGPq5yHkrG#m1h3UeWq*tj>y|R?rg05X@kgi%~OqGxJ%B*GpUBT3c2%KTEqNu#~5JY?%Ut0{>NHViEHOw2&bX9r|#|nl}=p28H*T(B)8I`-AioblC&z)0sw7sS?#p{d2yKnS0Uk{kGI^zV$3zeEBx=uGC!8gD}H3nk^ZZhU&Cbovs9+KLn-t=z4NcDVK?#WV@lh^7_yV+=pErn;AvW(o7Dj<#NrGS=mtDY;2doEvP%0o&aA^SBm?Zgb)Oui#aBB$-bAcDp2CG2oVBox!=&?7(_qCYG*fCSFKKGb~>Y)qse`d7_Y`uhcF%Cz-|lX4&{7HY{md2IHp4g$uNy?JcFAwaiOh21bJY4G(D0&CTL1tIc+}q@N}DoL&dh=z1%cf&E|x~x$obuhUaWw(1qxieoi4(kEWBB$ap16Ot;IX6qn0*#YTYUdNi1}=(u44nlxnX;HJU|rV|jkcPhpJ!Fm^D-|-6BGRBOKmg=Oo32~}-rKx##Uv$N?34O`w&AhdwkFG;sKej9%%v-x{*+CZgj#t;KJzK!vOvhQ37QA%fJ?g(?7fDsd{n)=!T_&qEo{ZXI^UQOcF>Uq2RnmAmVa@HlY?rTEwtJG_5uJmQ=3h@L+Kgm0G@I3rJ0;h&c!Hti=;!2KIFj4<2(~Vutd4Asd3wTB%n2^rqZD^dAy&gcvRK&#Ih>a!PHO=OP7aO^`P`#r`bFbT5VM}iG_5HuL4Z9Q=iXkp^}rG*;RyP)YC*Zj7Xz_BJ#V!?WvTRRY-%}PrQ@>LyRXpA5f46v&8hFn>eH*~i7)O}D_C*1FmR#KGeB_e`+mccHnF*h07GoqKwIX(jwy6i2Qlz<cq!JhMKI`t1X&Q_wcu&z_rX(Cxdfc9YkOcchG#a)kXGjvA*{?X<G-}_?2&R1u|dE*xV#Mfb`Jf<H-zk^AROEjgf$>mGN2l;{&r4_dmiLIUEq_Z8{#u+8dd?omFF1ZIq>6oaOn}WU6{vZ4ij(*7{34*^W}S9@f%;$1a<<=1R0RO(&ajCPD1z*H|ajRIiM!+B`^EG6V1t5uO!^{Vl*k@u+Mn2>we?rl{S7)C3s$GY7X1oPlTD5Nka0Iw8Rt!*b}%R>gROO*`~F>Yt;G&=??4Chk8VMG*F*@6ME8eX4q6>sIaPIP-u5BnAoDLZ84(VV$U8(kBAl!6j*Y7x3s;!dL>STuXflp6l_=>4@@4!jLnQQ*dS8ciblF-`s{=82nAa$R^DvPosH6=9U~<|8{8q4hcG$q+3>`8e-XXLk@!TpzYPt8c9>x2(ovMg>%bg!AV4VgF@kvn56#tA(tSxtzlFXMS_s!8<2nl2-Zc$#Q_i~izJ0mheDcfFXM4Od57swI@^0@*60?AxcmB1{WvmJ-rex#Rt!vr#u|3I*@P}<S1rrWK3|+%MYAoWHffFqU2F3N$n$m>&1#uB=Y}5avwe0>sm>J^+UsOFq2Uh>RUWcGtX!4`h@I2J6{@`HHDwElGc==}g>8A3yQQ{V<8t8iLPSP!{CY6La2-P`l5e&s~E42ef->NsDtsoFvQNczX4LzVqw+9@)i7m2q8)a>0f~^#@=<|@>F$oee1`hLBSyVR4<;vp0GabiKn}T@0(K73xQ|WqOXfxB<>ks=T02Dn==7Ts)!?)}`dFo-^I9d~t!l>SaX^F?L37uC67gZMZO1&~uztJ7g7F&u+Pyr83zk~G|O|-IDds3<3zZr9T0OnE>Pyt5I)4_Pi!5I>_If>rnozRN$X<kv|G92w-b{+9vn4__axKo31$mZOLb;FL@ub*w}0(`7qf8a7O@E&xeTNv<kL$(Z$*)0=mYR&zXnS4o|z$EJTt$@dMGi#s)G*{Y$wp)`XVk3ISm*pOu9}GP|L!i@qI2RL^qKul_5O8y|Gi{aI>-8j_i){dCP9s(HwFRKGek~(F9woXw4+D3gmyS+q<i^)5@er2|@21gkcyw$wm|4RF8PRzeOfN36d;R=>BUL8Esmf!7$%GN3`nXL9@6OgH{Y+*wom_BFWoa=twQ<Q_ef9tSV~j|#rYU;$#WH+dx-(|DnRT$tDOtMtZs%%)O5C2y?2F`uHX0I$J1=E@k8!1yRKcF7^0!GFEcv?~^|J(w*yJb!uC<Ryw^_>VeqfaZeI(je-A7YIO1DAts#WFbS2$$!C#@)iDBbrbqD*J9lqK@Vhf<c>o3tCtT&?2?L=+wmd0d|F%xKK?cYB6oj8^X@t)>^Ec8Ik}ir;LK;_%9Y0L5Ep+MF(y9f@n4+H^b__*iNx3`!*Ja5w{PVC*B!2<d}%P^8gqZt~@G+lk)f7#QEF;#ckPwj0XHvg%I$mzfFs3~6g*+J*e2v~{wJ5KNS?Z+ak_t>N%n9||V0GiJQL)#oD`A`9h9XjPg)<=;CS7@;*6#6R`pYfa@?{y>T#zd(lGz>7dT64aG<ci|r(>df0GFAlQ^p|6!<Y+_bYQ}cLN)!f4m<@=Us%8tQd=%+n;Lcy#i0^T3}l=q#8`aUas$~b-#V_<7(-|hG;ONKFHBI7ZPZ48!YBGR42hz){%Q)6M57R7<oIn=r&aKjB}I3`X9qnss{Z!>NBhHh4jQKYwCm-PiwBJ${-n)-zraEtTn2<HcnHL3{SuVd@Z7QAJhLxZkiwAAV2AVQV}(@r22F=et}HnJ_S)9~_E#I3BLCH|1nWyuPEzW&TK*y-6V4tPFnAvjxRBB*)ROz@ZB9aSXWO2>ZD&iWD%j6+F@BErOWuM%0Ub)=rwLa>84;!1F!81NWG|Ns7j95m7IA9a#*5h{XMfJKL1T^-Y{Mbp@nWe{h0%@A{R4a0dJ&>G(3{?!WoB>H_rsw@ysj5p;b$KEl$6XiOTY#X%g>ve5_SEQ%}dKlmw5+B>)-*?q|O+Y}f*7+5H0t5jVy*6esM_?H)*+}sTlNihJ$HsTUA=nL^J=5HI{>(b()VCFu#ZGVNgCK=h=^me<1a)kQW5$<TS#_i+A4S&FWS{A9F^Iug&*@B*h({X^+Hms@O;A#abft4WNa(GY19yolH2VpQo9|>6iC2WkvDjal?YtD*Z6s~z0M0GA=^`{In{Tl^9xz;HpNsaW3waF_OJese;69k7u?<1_(WR|pG^4jw9v+&a5;Ou{v|-pv<P0Wgv_c>T*m_5bgUGbhX*i+tGat={piJTub{m13kxDlts4>SQq#7IpuzjKEE8?#bsPY<Jfow#=nlAiOrvXjsXvzP1kfV7}J45rtT=*^rG`MvmL<4&Q?LXHU_nJnc+u1Riter?pd-qxhy&;Asqu~zK`}Xb)37=?05wk=VA=iSIS<qK4RnkHQ$S5FNWj{gx1ab1d8DkWEz<COY4LaA6?*wFZEP#Xaipbgpy38TqXaNUisY<eb&>szlgpl3Q#Ygt&IDp?~5-rXh4DB$q81$zh_k&nS!qfyZ9WA<}3uCmnfC3o73AFHZ5X)r>vE(R{pl}=q%lJAlm!hjebn5HO4VLeRkkKxG(+i_cSmtV)(Xp5+%|vAwq$Iqk#6lR$aokLDrsyz?amfTr_$ihUf!5)rQ#+K{HXO^r4gw-3Q#%?%Y3ry-xMR$tEW;!MdL0$=iOxSj`#^u>Ft}@fUO@|dG$Q#Gxdo^Xvypq{G>iysruC@jFUJIdnC6kl2U)5fk7h)YY`P$3o6K5KC`=s0M1-mlFX5=(5F&fy&>Dq{sc5fOFh<0xh;UR_i6=Pjo>KWFv?IT0EfSj<UZijV05z}-I~j)NBox-TGo3J1j=w9Koev}$F4*B2WU_SIQ<;Oihmx=e1N)rZ^)kdqI7db-&97EgECZdzQS#9>W@1DI+c2INoALZCzc($V3PxE<s9tMZ?YYqvI!wDfPv#1Fxh%tUG9x-DS;2-iBz;}$Zv{Oku~m4wynhYWEwFl-L;5l3Mq$`Z188z#0s3TQ{{Z`xx4=41r8pS;W8fCczKUa;93QjND=@DO1i<u?e-jKG(^w54(7zu4miHE2l3G*Hn`Ec;-70&{@Hfnph*)%hgl&O!ZVbzo^67m1SExZEb5c<SbM7&}_hGRq;bP+9p|Z^B6EiWKc=+=IA3UOioGzo)D>galmySK3o1JS7AX#jC7$y>pmo}D^Tv@?m!{xz>8nLa|zBIB}+0{IPVy9+>U>D68*t9_r(8NO)-r;6pUzt0mg6(>RwfeApZ@B99HfUvqkG9zu!`L&_c*zPHMJwIDW&*x#N2kMN5_S?@A8Jc9GzqyRUZBQ>P)7`klTfSYgLu#$n$T5swWuUQlq3MtY;x@4K^S{%A<I|H#+P}njlK#=l@k6NLYr%>Z$eTTB6c&0MWj50^|a!C`i8^f<BthI@yS6MBYj!4j_3+W0@g+I4qNV)dnNo1<hOf6h5bHz0BCy5*S?2FBEv2L?BW?zH}Z#ehW2Sg9y5ZaJ>OJb0-^vP138D(`u{S_=dkUhea0~vU|ft=(8!FPi#<?js|kYE)5%$!82>J|J7MYa87sdK_PpOm>;XOb8`fm2iB!iA!Vl4%49(!2tV+W<Oo=Ca=B!w}o*a-o`G`<e61QqN8lB9>MOc6hsPr-y9MePloSyTt{`MQ+5Nv3DPD?9W;S;4tM8}DBcIZlLK@Qzj9#o$Kod~0rM#$162-B^8GNQX%dV?E<z7$2*P@+%BZMA{l$b#R<f?vJDk%a?|1d(43AkE@M2UzmMW`4y)7M5AH0=Q?N$7rdMvj*VvD24I-jFg$p?*iZGtH9Un2Bvnb{;|b~%$kXywPQ0-L28~Al${fJX`v9Fvr^m7j*-3NJ)whMSaBo9Wr=~dqG${f0~>MhqgFJEhs0KU>vcc93W|)gRr1vmvRil0aY+l`MDVq!IL<&cJ>L=PT~afM{bF#w{D#c%uH)9ki8|wXR?Yj#U3NDCmQ-z;T$3fV;ujk1r-~1VpbGEX)b$<-F2F|{my?!UOC2O;h+%@!V-0?!HbmO`ikcdL|G=3yvATb<Od2E_{)V}mxU{i_)ZL`ZQKgD--@12+qzplxcrq)lm5X?c66)ZvXd)YCP}3kU-|x!j4L`dRb?VxcG15>Vk-(n|ZA&nN7(e4>XH~v;g@CVU*r~YB9<T5C)CsyH=3INMGsbiN`BK8?>hkaQbaQ0Lgvk&9l9wod{6s{^3JlMvf&9x|`7Tvs*b6ql{(5<OjgaSp*e$<!xm;VLY=>gmGzXi7u_2;?CmM;LY46Gl!vFM#HTV2}UEY6^_jA9oq(x9iY>q;0mrzv$SXJy&2z>e*%ZgxarC1{|ZF%$I^>c6H>$rUM3oZE0=Z)O=4c~&wf>{H*{H|Fux#y)r);5hx7ov=C%hVr2xb%95$?W+(%^%I-wvf-?V%EtS$&F|}*Qa5QQ#{8Ro|ix2pZS+1lV~4oa1Bk{cMefAeYmh6KG_fRE7UipBrD(SM&-pt_Qj{{i*(4$%kv$zMDJ!D##r3r@s~EA8L6p#8(VSD5HAQ(f)E>MQfpX65MzkeGM>Uz+9~DR#gY(tP@gck{LCl`5*0XS$QHv|S?^}xo0rw}B_hAIdE(rk`N@l5b!YDP4BhKu80H49LNP#oq6Jy6`;Fpuf*ZEOlvPn%0%QitI_wTX|6cIFpZMQ7{TuN6fZqrFK41X&Gi`TfQZ1}?y2Y(PKk9vMRg=%y>{c!Pyab9EW)NBt5fu?g5%CkR0U2%+)q*zcMy`Y}<FhH5<uPRY_r|TviR3%F!qPInjw^=^eifvz0{yC*zN+e1we(djiIZTQHgI#@-76!=JhY?1p>`|eC=2LliQ<pzvv#rT8~R4xc{13N9$`rjtE9)Gmm@1d;X9egM;K~fm{g?QMjAdo6_jMP0G}Q4&`Oe+=2AIPQtn&JitPAQAo?MJt|GS-A%Y?{jy2ypbk%x^E%iMU$%=_{;te88b=AnVz@ii?qit)Z@}x~`QaUeAnA$jDX)m|1n43TxRwRyj-t$_6(peso5UnFRJfNl!awk0pF?(^td%$`Z$UAYG$#PuRG8G_<7Ru8zZ8+doW!D}a47BSwqSI;W<x?QfwmfPa<$q@GaPVQZLqFa==!A#HW3f?O-1Q-<7W)vz73`Di{+8=D>j+mZbcCyxFYFesbZd(<E^U*xUa@}vGS{~2xYqgp^>5qk((V*5t#y(B;5=An%=-Pttt(f*c9Ye&$8}!mWtO?dPF#={LJgb$dLdtfv^{+;&sZ$iV~?&LyL9pIpx3wt7+&F+6&L$;_58-yw2?S=^$NBV{!G_!W=eggZct_8=2a}-LVHa0lUS<hk`}o<>3OrGlb;V2mPm#nVM~O)5SmHZl@*rT8bNLxo{7F3tCxJqpqyq?)Um9}_VQQd6}O*=#EWBN9QQ0yXeE_}9uL#^9$OcPG~QO&S$6^}_KtfE{CGsahPV}ukBe|4hqKpuJJ=?8RAm)iY3ps!<XYVa>yqTN2|<q{Hkev^%v6?lj7C;fYblkL8#^?M$?&00OlFXZqm9BoJDn50UX1qAg{u=L)Rv6#gcX8`+H*^i^W~$@B@kgG!LrGY7M~SrJ9%YEVNuo@xM3f}3d&LdSk^$B+89cEmPI4ikmQGL(ci(g$kKR<wK}WUf7t*0_u2nkTTOpm6@;yJbR8>xpOU`NWP;<AeK1B7*?#gS6P{G?ugmH&dQA}W554?yy?jm`c@Et>JhT2&YIEMQjE-ChL@ohnODn!qd3-HTv}@^)i`BfT;`lO6uTdN){r%AH{?P9JC$+oZUt9T2)s#s?wJTbx|4?}TP<Z}Oc>Yj${wEcl=u|Ao5kGHJ$#anEKjlSLUerV(TK}OD{h<;48)!tY&|FzbKI%Hf2k3{BR1=DyLJF8)b^2w~Wq)}MX-*fqoFTuNHgpM7p7dsO=Q)>#<vhAm-@l2EgX6jILkIgq2YW*uZ2cSRVC#iC*!mA0?9$a=`*u3mwM%udYd>_be=8kqUhVqf+v;2kwX5e9BP_I4t46bW-<ICpM63EebgI|WsQ&T#NoGdBNi<O?a3BibL>~wbzqT$ia-9f@ZlHw<<Xsvkv_zXg+=Y&({4K|k1PPU_gV2H{(M*#vgE$>$61o(yLZFchtE^t8IeL`D6NAS!P3B}Zp0@g;ZmX>$9+@Xolz6DrC^|G=6*yq*n^+tN6$7P_t%`mRd0hQu65{b#DVM887oqiHQ0z$@f<+R1-vrI!v0;w<gFPipWHKE!lL$7#e?54Ha6HqYo|<%5E0^opK&Ef^cD445el>nixE-3}vZ#2h7m+`+w){PbuM+`G>zt$3)M$mW+T>`~Hj26rqO<+r=xmFl)S<zjtzdPL4ptYj{3&_e)31BJ&v%KrOd#q*$15})-`vzEtjQ>dakn&vnV_csXnearX@tA)8RJgJt@@TYDV8*(eC#7W_Bx<3GLjP7Ok*F@f`ul^PWK{eA;<$K4r;iHNSy8@C<KB<;459|6V8nhF%Fm@Jw`lAu?EN=Mz+#BUbR>`FHNS&G9<5&QNC-$!&><S@ntpgu98$EO$omI)yp{VpAz>*oA<EN@;8CUu|P@8<&q-wyYS^LcKE7b$v_{}C>4r3&3I0XYnqEWZgsm8W2y9N^|W<7LO=)NZFf{WAAI^WBt*5rp|3e1daMGV{}yBfY1WA(DymO70WE@Ut&2Fhw!2HT1ftp|g*YMy0@^xny`WO9>xzijQRVLeE-QNEbRA!AGS$o^tyrzY_Q-mQ+ao81#tkRqdDL94NZ|y%xVU7kdaWR|G%rn3=2F))$dtuih4B@q1P;3eMk?xB9nrx&9ZZ+@B%hlM64&T3y*~MFeSDn}uwrpp|M*zYySh^`xIen>y>-!hejNMzkAWLzovHfx+`i<k*N=k73>@0B?aDyGr!AeA#k<?;bo>L8pX~6UerRViZ#j9-TTb5dmXr4!qR#}j0oa8%WpZT{1{#jT>9`z)oUEP*5bCYs+%tMvA%N9}PS&cHPuJ_w#!0YVCSI}(2?PJ5tm>cdRB-XirFN(GrKCRJsnf4D`fr{7dq8jQY?O50y8AEo2xzBPdbhP8^MLbm*e`<Yi)!{o?a)se!xy^#S{2{?3X)&d<X6qNd#9U}7uQ^g%iy8<10Dbm!TL=Ag8s{b!)L(5orfsJMFW6HQ2pL8V!a5mZ3Njif@~YX9|vc(A8-aZtJc2;&IIdZpzJe)*_{BT0~8O|81~lx3b1FrsAk)!X4|M{+o=9=@LT@@zkuIb_1nU)Mv__R{mkHdCqRt=Rf30r>1yTg0@K!uTDFZ^wvAf0joKe~F4lgW3!IC3<@=utjnuQJ;xiwMJ5_ESb*eQ!EA<~|<@#sEdQs1|QO~wf&$dzj<IdaqkMqXoZR0<4-t;AuJ#C-)xZSCuP8D^k%pz7dew@KS_6%At*0OD^W!qTGwz2ldo#O{T&N0sM+WMd89P3Lndx}5vF}_np9mHL}Q)Aw~R{e3d|9WTJda<5uV?Eo(dbW-AKMp@^{J;;u59{?mH-6BUd=@)=X7q5UhB`IWsj*C^w*CWW{5^4o_2NOcjR)B_9%S2i@ICMjl)KP1iv^5U4NIQT#L|ZSy!ntqal1u$;B1BruAiW|)}}Qux0p>@Lv<oP?tX!ScT3y3@F&(`v+aS-9&F5Lw8B<JqgAb}Rh`i!q_@+_V4;Ssi_vVV4kE&8C-k4BOrphGS38Hwd%g3E*7UZ9qo@xnr9FDZ`3B|9TfabMb}h>nSVGLEVRLRmHY9NxmbwL&mN>!1zdQMqvhr71$i9-r_SUyx*-fB0ZLG;%wGdmQO?7b6R&;RI>t<=Oo`)F?y7JgCN_7A%P^CvmfY4F@CQwj!)dP6X0-VCJl6*kAct^qLAeJsq#MTOLsWK<w9rb#2VITu_JYAk0uOLvw0!BiKamYd}_>!;NKuJ7(osUw6BPl}(l4PU{HEACn^k8ZpMpHzF!|jE2xqyMibTrdJwh4?m=q{^kp=2KrAlv<M4kETmmr%aF(XeY$4x9E>s|SnbTsv&>Az6Oe+(WEC%t;bMFAqUS!dV>pmt<}482`wQaq;MM3St#t;X3Sfmv>g3S8m`rX<R4aa@1kZ$(nJTbo}|84uYF>AU(jS_eR|yv~lJ4hBnU6icQRmBN?No;CJ`5WO`12m-)k9?s8>vaqC>)jMqmxA=m2<`<wE%4S4k0I+6Dg!L81Bf|TH^LA9d1c-GeGwk4nIAnFWf-4I!yVU`#oJ9pY=lEgZO6=U_^538yV5KdbqOfwvugk~TOG3xBA-&B8PLro{G^KeKMCtp;OrM%dq^3_8)c2ySZPd6&~ZzgA5CpJgVCGOuZu095ch|{Nb{j~F1$KoIti-%lgQK?rrvo*B<ix{@S7Av0(Uu<D2mEB$ZH{-WN0Mht;`FDv+=s3811c8ho_<kdZ`#MrW((4%N79{&y1eKsN6KB}vg=*DWb+)|qAX|}WVEpH_zPylW-T3P6zIk5Bx$0JY`};<|z=m5Q$Q20O7V-r)0s})!d>BjWjGoWv`HY^=5)UW%GkSOD1WNv;PK~-;kd$+qOUi8@#B^h67eYio-O_7Pq5IeU80e2!kYGsz)wF?Xu7R4>K=BCJaWNT1;GJbhAqA3=P>JA@hN@{p)xw5qRzn;sK%5~l0qggM{Seq60?S7twM6?G9Ge2Xu*`7ea?KPMZN}@s=D|bUTk(`{uw)&G?*~b`xe0zAq+Kbu%cb1bmU0_e%I#i>hO@Mnpj5b%rNZqi6>e6kaEGj3lZ7^xc&Xb6Ey&|7cwd->)Yfv3g2$efiHok&^8t)+nSA>xLpVty1Vbq?>2}(>Kmwsjh~QY$FuJ}=?1pxUWlHYfTRb<SZfQ;@4C4QG-vqfg)!dufVZ-qfF>|auE(E&kjmjC2GZtqU_6Pt}B3%&2B!FwP6^p-|$4%cA-auBq)g{_>rBbf+v{m&L9OI^{+Kv>kkY>VW0bWZyGnf&KktU941>p)iBciK0njU0FgH8*HZr;4vj=z>+`_FM}U(2fft_<N{P4H@Z*CDDm{A`+%ja%Q2?zPF@e7EB1^fheGcj9I83-X8XHfBNcF=z^jxxFV(y-pg3rJ@6|Fd6H0(f~0jWp~{gIxYF-%U}_HnRWx+Zcsduo76BP(JPb562sRv`>S`X;4*~)qhbb<C9~%MHSy7sib1BLHP}ll$|mVxHWeqDO~nppQ_04&S-#G?vA}68up0|<!Ew^yHpNJg!#2SWu8&wv@0z>&hK}J7BOJGBdIjmEMI!W<7KurxaeCuh(K{fl0cLCk^2-n{64bL5Ton)CNcuT4-5eGBh{jd+ZM=s8S>E;62u$LT@zu*>lh9<2G)XKR1WJNOY3NUZ^(oNYiTPHwzE$07HS1H&R4Y7+Id<am@dkzCMLJaZlpk;UEjQluQ+~XGn`emwn_Rl9k*2&OP~JoYU}s$~a~t`#&#x0F2LT0ipoJ8wHy9+WMX`gpR1PBWV=YZVnO{NjtD5`*@!LKJ&rbhdT(UiO`+;shNZJpQ_JgGTptyZ|IhPGPNQNCG!w!;R2T8M^_}}@BhaDut4w7L9$*>FCccCC9k%`*Od8ulD(VdsiuSW!mqueDyma~YiJ?rT}$GgU#HU9Pij;}$^akl2mlQkCvghptIEYQ2f2<&N&iF{l<aeHmP9(354wW7$Z9G4~6wp;44rvuW!+1QRqlTJPLM}D<O4Ze6v)b6W2;WRiY9j`gC1=t5(-9{47nU=vg+s-)M^KLHka?<{XoxLKzVeW|Fc!uZ$t(f>0uGXR&UVJp@VQ;9FR#%vs?%;gT4bza)Gl@*V$&-&q?dGUwB~PBsbo}jO{TI%|b^zF81X`HQXv#FYhGNfVdR)bad-Cp>H=@UCw1eX_B%?_>WFN6@tHU%RyTruZR1zjdLfrN+LTb?X@ZKs49h!;SaCO|!50}Rc%|?az;^dni#7%Mac4na66zNbS4gzq^$LDWezJ)J8hko@u69aWcJz#38=41PllJ+7oN7`K2eWVWQ_`4{|aeYa+o@~h5Xv#52Q6m{nt_&F9EiQV`gl)|oC?=^8b7R5rn>^<I^gWcz#S1oiOWLKPbaw71`UehV#S7~l`;Nfh(b;tq<*L!57Hax7)5McwNbSMxBLwCSGxflvz2TrUUF}W6FcE{gh-kq|rK|B!8fU|aMoi!+@rY^8__4A@_kA%9H{p!3h(Sf%`w>6b!Ka+?;h+d65=bTcJ+n3Oq~pe7!GV#$0#+a*C+>`QPIzX*m^leZbQ06ASKx7JX!p|Sy*e<MX>egThGq!b!)PjTT_%Sb+|ja$Y_q)K&RDEy=vKZGRb!V{pvcudBqGTP7^Lcku%+}F`dnysC!*OE-4M96NvEXFvoTQIC`mWl*mO(pCoo>(IIjp#KFL8o{`Vi>+wZltOLvy}u!+rwq3E*xxTR~`Qr9m!lsYi(NjQnZ;U@NERY)3b3&7C@y4{JSCD76|x)!5O$k>RjL%fy_Rx*l)7a}Tz6!t`G^8H)O7)l!K!N#GZrC)U8Y3|*5YiL5?F!GMf6C-<q?y-yOzY4VrlJ#8N?rZ+Q1wDIUN`{5nkC_DgnWGEQmUwu#%I}CM+Bfly0QH-wdl|o6AotA&Rno@_kb|hj`}t^c4vNfS$P9kehZC1><~W(7!W(yTc*Eo305cyEyPcaK6Q@0f8*VChD4cZbzHHiWTPm0ViJcTtD4Kj2TQF5XINbCuzu7)pgT~uM<mZmRo^vT##{_i{(D5{B(M~upqtV9};ec<PD+JT&2?2|&CyZm5BN&h3!4#QHbSM%N=tM^wF?xiGIpt@G^^9$tqz!gNR}fBgVIuElZS*emxHl`wtLrLN5DrO>R}$bTiTf!b8cQ71=p+fZ>3N7ecY^5iRN2qO-t6(UYL_Uflsv;gg^s5uvjreze|`7<8Hiy)Y517Hq6ghNXl#AE$ZL;gDId_JknIQ>n#X`MU?-%|=}!jTr-M^rsr2eGLd6WD6rC}H6(K$;&CV=qMC%%H!5ndFg|jT(jFiTxuOz0Y9j2wJtbmkO8$pw_nktWoKmpq`MwA=0hver_W33_xPT2KtW$>LaBFWBVd!4kw@XnGrjgpyQ(hfL!BgYoZgv_sGG&?L8v0q(|2dqRKrjn%NxOpJo&oZ3Bp`b;?P?ga*Z@y&kP^=q@aVTDKa+{Ob<rFPIbrXBVIy(o|^HaE$*T6_;5?wD%wcy0ZO<?VFayAIalA{o1&Vt*V715C!%;34g(49-ij6?O6qfT0i_<Y`cu(uM5FV%tPu61BY#xV#H1xn;haE&RMrm|psghUp0B8)RoA;4W2m6N7}PQ%yU1d=q$1gu-=)KsuMlR!|$2DB)IvqAMbQZt=$Z8wSY=bBmEI1k5CS~R)6u~}M0RK$*3E0n?#vDch?h|p9f^e$vF!DhyK82h<b(i&$dmj$tiqF4+*#2`cRozb$iX}wMw*@mmRhO6JE;aaZY+P7)Ao@==NZ5m$7HN5t18eY#ey#8$(evoVU!B;oj4STKGaGGm*<11TU8vpnvfPYP>zb3R_6V$H>;@5=lYr^$4f%=-jd`(EcCInv-e6I<%*M!;Y!s;~v^%4MGa;2E80IjI2<E_dSWAXze5%L#jjZ}7O@*Jk|SfSM5ba5r|RvGB9i#lj4hff?uVeYCv0cFiZg>^qO1rizRBgc7!n&*g?$M{d~b=*e|Z|x8#OPHF7xta)Fs8tN&8XXy0IdapRfg;zr_`u)sn#_nAPka{W;+MdpHVtmaO*a8q5j{OT<e038Q6C=qZXk6lEcFeP#Wpn^O?`(5<O0lExG0P!MD*Mk><e==rA$En67vAsIG+Qs3yeY!dWp{TdJPBph6B^EtK&5wUl^b}uZ+nt{6)d7O9{1`VEpgA*e1Yyh1JCI8SFk9nKj$F#%4%7{mgEl!OE3evU80s$u=LD<^wlO{aHoZR$?ZV!v?<zk~aa{+oUP9Qqvatn%iM&FK!a8J?n(c7B=dVVa6~M@8@FOYd(*~?XUJ64qxj~Erbfy>nsvI<Z)zll<mj_i#J?JMRO%zFEer=QY1&<ZSq0SuZ&q8jg}bzG^4>cu>nY1d<0tr4p?oHv`95nnQ;Vp8Zev7GV&GsWoT(vw{`kwEg7Q|#h9}dt9Ueu@E2vFx}CPb0fh@fqj1WR%)i`_7N@baBldIorIBLzVp*(iJ#9v!wD`N*d;3B2m$&M--?Tz?FNqwRnqEtqUJiw<=HVoiS#{yKCkh0KS4><Jn)6HMQl!=}MS!n110KzzIEiSSLc5VJ>~8Px?QieDd?KkY5tm=`wTmO2iN}nb+j>yA)~V5KZ1prJK`tz}6<<G<d8Kv)U)=cPdpaHT9vw0y&7|wHdl5B}{o1qNms0VXAw0x$f6NOb`-Tj-Ew~S+(VuBbm<$q#r|J?xiIeB-3)2%Tgq{~5Esl8k`&MU}%*RM>FdI^xX5WjD3@EXru!cxT?K$VhSYBGBAa0GXFgw{R<AgNF(}~F|%{n6&x=IGH+81feYKEJ5B!4TA^UbEzsgG5Rh_*Wlhg~%rH&PHtPSqfiuB?pf(u47<jEy6vZj{|GMdDy5s5T+H<8X40Jv-7CB1|2IwE9qtar$iwa&(pnfc3NO_pkr;wq){$`%7ZRvx*3DqKhKO{|@B3B+su&^K0dtdgS1;O#10Th9XQXg$4xad*vA@?1RB!nk87@IqpG=aFU76i$wR8hH2OMkY}R~&N(p+r*1|<ijIUwk9MuY-ydPSXw4_1_CVu-&2-lkA(a^Cig%}V8miM#H-rU5XB1BpO5o$(w+U`eP`Mh9!_J^JWUqJ|z0=i3Kijjxu<N6@eHsx_Ae!ajy0M%C+^W1=-PW{KG9_pWXyD+zcrM^%fbWhpFEf~!+-8bc8EWYqnspSVGcgV)=#o#32DF$UtXzJ(zk<Bg`W(oz?Z4UP%oX-M)EW*mBh`cH!Zq5Ag^-tNMASLAEQWxOQ#8V^tj-CCDV<qB#n?yf^bgfDAF$zQH0Fr&$<hC~xAXdy-gS0OaVKqZV2DpPYnJTs{=^r98B4sWn|n-u-@Yd%wAB?K6{?6{O{;q6W>GjPsb4x%Rb4mGg^F8MB5=8-ts0JdZ_rmqGr~u=1mQR7P7Wti?;iqWxLEbio!7}N%L93&ZJ4OhlDeI~Kb_Ikf!V@EC#2Ojg>GBUkMi#QuBu!edpy2^77!}uOcOoZ@)vPp@V8wl6`_3psvIKRwS{+^^Z1GZoa-KumZzY{OzMzW8`%-P`0WieQ&)bX`nQPmhx$w;WOBEU#;r5DU+AmHNnr?@;0~=`zS(}7nhTzYqk*keij7s`d$CG<?_@LK4LIJ_B*$*%!v|nkxk217oV<y1I2r4Sy2<ML_Q{Jwqnp(k$5&bXFC@mW9~*tpW&g9!q>!;YMtPg?{>G5fWDctMwk5|fV&{s=42c5SU4lKdbKQ7wE>Dy85YB1MQ{Qy|JEpGBrl03YJhF}}r*NcOn)YE;VH;4QI2%1QGUrsnL)+MCU#LW48luXORft2wQd)_y4eKVmV)D%QW2Ar73Rb#t^-`w#94WZW+TEO__-Hhl5+SA&*ZZ6!EzJ<Acjeo(YNee`vrQhWe{3-fT5Bc{dHfg6Y@ba}ZzWR8!CRuGvoqEgW}MF|>g+`IKT^@(EEWBa1oGb@f&5=H0sP-7f&1kW+pCguU&{mT%Ea7Xou=J(LGw$l+gbWtbwAgV&sQbrP7-Qg&p7XQWXZnIe8Og)F-dqCB4mlT1K&fG6j)coAEg$yj984W3RGw&nKk0>h}~}GmZQ6h^E{)xa=9m%Sfm`IzTxPS?Ht*xou%RFiS7%7&RlWrwlPF`zmvV+_R8~^*!vYK<lb;%i}HR)?l*)>@U55+dQ?Pmrfb06%iw9%lBf_^uB#G_>&(yXb~Z)%*XAF;RtnT;n-LT(1oQ!``X{=13t|)$jI0wJCe7#Sp`*Pyvs0G8^;70fD3voSPdPZbv<uT($*^x$Hfb;6Pl?7A>UY4(0Z_UL0Zb^G_fs+#1~SwhZz*<6gZt~vlvNPM9TC)Y96<*DO1b=MHOQq*T~2lDcSz9wIe`oj&!ydLri?y8%_J-_2i*>j23qecphR^zA79o+k`vL|gneT<Z8sw#Ef4DjOa}_Oe1`A<llC(HAlp<w>n51>6V!wfk#naR5YE#>S6CvOJnCs0^(+wUuBMmJLxo4pI%fv%;!{YNi6bvB8hE&nhYNoABoCkXVJ;8zVv_rYiPo>o!=2YogBf$w&~_1F#j;UUm_w6{c?5>@pty+18hH7xT;k52$r+zd6~8p0_<C_daqkF388%~+H;6`WUd`pzC*MssetEL-b)9UyFxU9H&NZGmBV0Gp__8$PU!P?BTcj8_oaBwui{Ci8_{~#`e{Ev%FUwMff4TJ^l0s%Ae+5joT$*%yZ2L>~9WBh`Y`F<cqVR}CUcz<}R%5WcH*K+PIQ6)S9<SS^yMAW9<i@&W!LG3)M(}AVnv4U`$GztN{POaxv6IpoFi|_l+FWW7>x*{Mia2fTBxL*D3#XG-B_V9Y``@B-cty4>@}r%8m55Xs0;z=?82g?QMQyxlSNux_rLLsNMWSb1FwsUe)S0F+h>W6xfTy=Wu4j3IO6Vwz2QZk>PB*xs)1q%V4vnU55@%;3b9_RwZ)uK3U&y2nFBQ+5aPQm3t;~EXGK;rIdf=Z^8>vJ&KwbR9>MmIxnY2qu3Teem#~B-|ss4!LdlG{nZwU;TonRk8;R|fGCRGm&ra3coY&gjO^l)BF(50>-aB~~uk_cKe!B_zV?RF8Z#`1XQ+=2&XoRD`5Sth{0RJNQY`6g{nCZ0&0Z@@$K3V0}oh4TNtWMN*in-L%9`Ij)^Ul1%-IAZv=t5&}?hzv&^pKh{$5gqoDj$v1wPC*{!Uakb`>y?ed($4FZ!2ddu*0PGs(_})ziali~JWL7MZxTak3E8=vM7x9}eekVF(%ceEGlb^@NY!TJ+}jJsj8Bt-EWxsTU5ki8_Hkuag3UG|jcUqqs~WR1Z#)^b!{(W{_wx0J=F^w|$LmjtG%MpYH=E&N-1AoZQ}&phjeXx=Zr&{P%obmb>`!c1;^eB9Xdv6LSAk+=Q*{thamt?A!jWr{&+(`o|76@LV@p3$XiZwIILT{ha@)Wdv6rAkLag_-#nkaIdF%1D&+gGFM$JIwG3$zY4g7dSzlH$K<72l^hc02<xA1UDcvNW>ooQAZ#93RL=pYM9@?G4ubeV?Ja-qTOG1?)D#VPw`$^K=$D2k8jj(b#dVx(fuwU{wA09X~aQ>*C?B2GYISktA2z~3DD-y2&j#R(hV4kujBsBT=wsBUB#)$6mWmtn%MXIH-s!)gQTjhWWp!M6S$#`Wu2S2wm;A(G4RLm9z$Jcaq~sFz?`gI|?pnG0)4Bf}@APV1bqmNGjppO;jz4RZnGtZiuG@sLPBF-djJlb_3<jy8zVW;A!%ith6bOiG_;7@IT4HguO$7CqO7qa2NBm(U&}S3OU{C-h)phu8^+9_L%(e+d3oAFK5Fv!SZQl7wyNdV>yI=V5`drb+&pL8KQzOvJ*o+2Fi24B5;?hrI$@A~tt&yMad~K4lK`t#~A1p<B}d!cQrFoh7(>K)eWs^0RhDVN2vEzwGVrfWl9oSb6D|*mFwwwlT3f(H4*CTKGN0r8r3}Akzg;muDY5Vp=2OD7on~f?V)2d)l$s`RYAKq#QpjV{|E_DQ9fp-FdDSYDnM>M4t9L2lEXBRI9n?na&z~4yoiLT=iUoCt|g-b7?E4g&p0(juX%2qZM`o0dQTgG0TcDs|fbtEG!2jr2%bb*J{3<Ss^N^muSAzDBgmPI3*vw=ZT$e#AfI~JoM>tO6Gtyz`0c85yq@-Ru`K%ZKH`7!8rV*w*Xc1@wpFR*72Sh#_5us4O@JVwnx(=6~XGMJ(|I`Efx}ujbdYok$pf1JKFQ}UQ|ptSch7lXW$9_^W$?3nG)oLhnz0;f>#>9Y-j%k80P0D!XEXUr$o-BU%t2@-JX0hee<O6DeDT>&74CNw?}(-PFrQ#)Jk@?T^Bq@^~5+`Kk69283HPA?A|dbs!?Fu-VrN?=ZPshTOcR)z|X&pK`^zmsow9cinBX5S8fuxbT9+RQvzQHN0QZ7Y#96@;vMm*P)SwS)>o4d>i%Xz&?AQ)aUonr<35GFbs<qJpat1akt6?63+h!_?XlzHmJZiW@3mcB=@w@tnn+C$^b$({jFF{|2)9Yx4Aqh>F9&^SOKqKLUCl?6ZhV+#|H5>W3}A?@K(*O8l1+l`7W#*X3QG)Z>Rw4md_h>7?m_XDph^A|Yc3h1DT&E95-^RyzYq((kZHn5P}X{flZ7cx7S_?dB2^fzTP|CepO_%y5A1{?K`N_{)?&0eof%Fjq8C^T!)DghZ<-oSAyMJmlVBZPdf7B-CQpu?5Dl_TM3s^VCb6HIQk&r<gomdS;@nq#4o{2gtq)@2K3GMzm;N}KoUnTWAMEx8Bl=a@Y}n{A7-$Kum%trD{qd#RnL$xHoreg`KWA(0I1T3ISeUaku!TOwrPGK^2#<8bMe4Z0whY*?*5txZTyM3j1pPxh)FxCkkvZd=nQz%Oaelt{_E%NA$~bi)a&|~3<6CYZP-+1pPC5jQ<@IH$tVWc_XQsijpl;SXOLi0+VehJ6)bp=uE1gK=@{tp^=HA%j!9E${2BU*#aJGxWzCbyo4w(7Ah!r*fC*e)$kQKD^?&Y&5&#0+~)nL6*v|6~l5~T4{oLKA9V(55q1+D&yJ<!9vr|Ol5PTNQ{5itfBO*i!s=czF8>yw^ob?0MqSE3iuFATUwzkcO+>--Dvz#+><Xc3<_upxM59O2sRb0t#&R0h(0-i8;Ot{1}wKR>y>*T3xVv7vaSQ3znY!Gb0wy9^vI=-Z<kmRSbeZUxQmzWnv0y=rSgwc$34%T`+S#B)B`7dqYtve^>aQ+t3Q5RuiBaEhigES+H<!KkNGLsEi!>6pal)cs6zsI5zIeIDBib4T0pWeg6Fp9ryj+@uxCYq~fpUt6dxO+ch<t{&*KH`~7&2G_hBw=Wqz)NPlHi6bv#M8sZRYeYGbH(Ge?U2CA1tuijr{6sK%@J``m)x<G~8RwgmO*~y0JiehCfP4+L;7vlflbwgcDwP<-*VYKlA&^1DtpasRdH5v4kqZ5_WA$*oT&8naQd8QjX|+7QAQ)pe8FdnksDQ?E+gg=FKX>?=Lx`i+Y8=u{!g*ad`CH7yM;6&{s{q+~gW4MXQ>UXHtkF#$JfNc;Y|y)ErGkGR2vfs-Q|s+f^Yp>P4b0#?*r+a)XWVw$F;Zw$%5|#SjybGprTh@ag)vYHJp>Q#_}NhqS%rRr3Ka~H9ymbm-vAkXgF4<}6@mwK>mVhnSc?@R7#g=aCmUW$F?)AA{=tUM?NU%@V)&sVF9#hEj0*h*L;YNB;%(rDcNbgAaaui)7y~AC4Xgru@`&ciwl-%q+wi3b%RSj~OYJzN3TDH%$u?)fGD{)75qR>qo&2{Q8m^0!^#Ch}DG$ak9vnj8NV{lW?PSaBBxN)f=d&UT?oAZ{oY#QRFUJOg48Xq1;V&62je`a75$23^Vt+Q1+te!{w#^(xG#6bILT}F4M`M`{#TXz1j_;rS^8P8&(`;b}?!p{xcDjy*5Cy$3owNo+c7WA1n>Ha0L!JYZhc2)`-`;<*Lq{}edZF_#wx%$m>+;poZgHR=_(o`lSlFBrjlwdqn0PJlVW0}YwZLkhQAc(f@7~*UkWehHbGX(f$5zx8O|^>GhtOVZ20Afg%=Yqh_vKr<osxbGxvPm|5fM>^L3||SY74o|333@Vi4$tFV!h!kJ|ZmF9!x7{)k}B^3`)&*zx(8+s%2ji<XIy+VO9;fUD@E4$vTMwr!$&)bvPP}KdG$6+pU0*FQKyY7QKh?i&pvJdP;k4swWSKI`FSH!=YKRGTCvav(f@fkWGPc+5v%aac84Ppc2PvrKBq`+~!ktrTOgl-n`jflIk9x9tAf|cV9{J-FN*1A(1i}j_p+UPNsV&(fu{^++Urkej}f#K9ilE^TIp%>Ay&V`ih+r{&wC$U{l@R>fQFbX<GQ?F>oKN?qlr`GRxaZuJ9yPc#<hx5`~v?`N40p75_y$5nR0!EACFLxI3}p?!?O1@5GnOj!pG1e{6#FD|d_j%kI@@Y}TEJD8)qs>>o(S?|GDx$H1){xOD@!Zt!<KhTnK+zvt0R9s{>-;MNV?y1{ol*0sO<v98u{y!nEGGh6?e_x?_R(gBJGYYY(Uf8L{>JXYPhRkv={ty}%OqKQ8MQvR%{BYCX4b*paOs#~}E-Ox_`FOPO=)o%by`ohVA(r1RHI{|70s1iH`fK@C1ov0{zthse-Zrz$&xAu2MWB(z@`)@>T$z#o}TXXBy+`6^zh8EZU@@TPM`A)#8FW4+-erDLbQ{~oCr&?oxt^YukSyY)k*4?^ww{G37TmQSF**_5^|3JMi>P;T&Zr!?Dx9--he>b$f{?|v_8-F0MYl0#R>Yo|b?^IEziaJ%6l~y-?pz?2t%9F=6x9*x-cg?N4_P4zT{@f?y$JOxdu7>1s&8@rU)?IV!u6?&_<-uS6T3K8F4<8;)KxL24XFfZ3s;E;#of-@OYt<iD&7b9}NgmhTy6bM;b+_*N-}RdM3!bzeSJ$88>PjBh-MZ^;-F3I_`ggnbHvaP0-g^B{co;Rom_3T0`7GY4p-v5TYOKtvt^c?R{~NBt<ne)9_kmmYfm`>%-}M^(FFCnCuGYW))tWp$aO*yB>ppPnKKROO8Qu=_k!vm;ixIW)zLGRaZ0&5K_8R{&@&BA2O-8f+kyS~jI@y1-tutg)IGQjFvpL@~-2B+cI@_Re)ns=-sUS_6w9oAG$B8nvopecOQi9!Z%-#*Dm#0aL#<tl!SMB|5riKJrl5<JB7tW7HL!Il11{P`a@P)LF2Ht54d3G;s>A*O}=JC5P-|y{fgZ3mT)TpQ921I>!@?)>o&BV2FdPLZf#(+FSR@xhxqH)8;!p#RU2h8gni6!?b`|_2H0lHXsi<LgdaBjCcCu${eh7{Myaq9xEm?~V1N3o;~OY+Haww<V3R`1yj0haQi4o0CzGi$HPG{jqAQys+PlIjP(O!=UX0NZ0<dGQ$GHTsoJWbB!QJY_^xE9=$V;jq<7x*YzlJTa%Y=AC3R()RK{nW#f%8U6Bg94^v$7slhFuVX>>gKUt)8JwIwudhQ0J$|g}?Eg}4Xy?0RcqVBVFF7`lBOV@UKLg`xDL#y;Ylq{9arZ8~vP>Wnbb40>Yd(@&@EaP;V;U<$`q9r#`k8S!OYHE)w_$mTKJBp>4o7q4MPb-A-lvHZi(lC8+@Oiy6FTxnQ6U{44Ldd@0tk57pp&vjzLjf_H#-(5&d9hvrD6HU5?Lq{en<tje-!ot4pQ`2haWQTc}|c<9fPtz;aKqF_92cy<{ig}$6?T3NdkfZKY(68@Uy}CW}N7SJ$~x6(`&M1baJcO37_5g1^Tt;{_^x49132yHZIS1VoroR-;N9CpY;$!ydw^mAg*G36bSO*Fjk^}`Z4~Ed?&&KYHaaw;<_k~@hA8}qK7$HA{PPz2}7wij+_ovM^88q`DhV{-Zeu>TDFqTEJ{T^j{_moOL2y>gVl11Qt{ySQ`eOwb}IY*t|&wY0Id9*sjEr_QI7g?(vain!dsqJ-{tC=91oeaDieg);N7ufD|w!d!j}wwh5q;^DpxGQ+)VLoB9<%#e5(D6%MzThOsGEWOq<h%o#e0FUazZWNgdoll+`<dPWlJBA*OTPoYksu$?-KUk@m@iN17PWM7(%ieR0cnzecB%sdkj&t3AfGPZ#BSeN{p+AY@!ZWF-0Z%stJ4K{cQQa7LWm0A6D=Nnw|EhX`ZT@aw7?kBF4Gp#I{i&Z-E6HeJ<Ndjy~IL7VSwAs6_f?9h0RXqeQ1cf_$*qTxjzJVTBdr+12QjDD0bZ=Aj}pWlO-6&e-aDaXd4P}%^59^%x5h&%V&Z^&wdP#2H5>9p92Rm|7%f}VbO_wJY1i2bt&s=F2Yf3xAR7%mJwRc_CQCn+opXYl$eOALmx0Ur7=Ivo$gDM8#~bF+Oh4de3bD!m5%(d1<k*g3cZL<h#{;7$#{>ctF+%7T{;XFQLcaK^qSTIs0dVNYJ<0F;W;8fPKXN-gt^7veU7%8TaShuz(mzpC4X>mWsbdH3x7_S>iG_8+?>-1qkT=h+3Zvep12uB^-%Ya68aS>K-x!GK#TXgHD1*jYUFv)p7iO{}4$d&FUtOWe~8z@|l*8#3RJ>r!&Sxj3eS{tRh1h^kv`n^;`LN6aUK=`@VkXZ*rRr7Elh=PXXLbAlxGku7+!lf?5%+A!3<;SZ7o543=9n`6QP3(l^zQxtCE{VyL8k%67=S2)(n|Ikml;ixJi$w)Al9w)j4Vnb9=_IE|#l0wRA@iml{PF#_ts0`I4ac@%hR=Ek_({5r`JfOzFwK*AUdBbQ+2Hjzp_oBUJG7(?LmDZxz7rETtGzgZ<D9Al3=IV)VdvEt?eUY@bAhUM?2`EuQE%Fy@#}%^SMgLGrG_z|9lh#-cw+<qcO{m7fvzC&ti@zNCYJJZpMHeba6Q_$ErcD<*OskVyl2!sIb4zVp(%dm-jW+)qY1?7~_SZrJR^rVpXa);Iz*dDMaB0{j`4jUQ6qQ+ircV(;+732Aq^gz1E7=7rAPH3>w1{emRN}_p{0D76BFY<n_*Ytt4S!iu#)V)tos|)za<eiL%Al*t3bo0>v31;mOweN!EK=?0gHFpX35_K%b7wT1oklT7gP>34vRbLGt;tnZb>&M4Y*$ai<ix%RbLNxLl(FKewc<CLuSC1Sd@UXTQ9FvJeI;x1E3+nxcv<>6kTg!ZPTqD75j6YbSPULU4)cEp>aCDoyC#9Q&`5j{>sUI!Ce+z++X8)B-+M|gBLR{##X^d$VA?{WEm9KgSAfexlW2?OB${>gn%mfu1l)2*oA^z_A8Deh#2i=2rKUEWYmwtwXrY(H`?iQ5bWCL604W>5?kzZccoJDiLQ@P|7j*M#0jUHFH^nvGf)=sFOi!UdT}o1LX=1KeR3%egE&{tcvvX9)(WO=13;zeC79rvwaLIcFt;K!&?`nOM*xpvhgwm9Fkr=p=XquYr2yYNa7ZJUl4iJ|*6HnKPR{6dp@g4Ovl1e(j?pd(3utc$RD;R3z*ia$h=m~r{%p1pV0pm;vHXp=AAP0e5ggIEaQCtXB#{9k;j)@rwsgoBwZzLL{nob5|<{_wxbjn0*6fUNC1$pWS%=(BX!oYAkB9gY%#=lJ2o8)~`n1{hw`$C<zE@%dW7S3K!uB|>$GiDn&a8shpG%<oWJlY0M-HWGmRxw_Lc_eCX7M<`2b&^Cq*xTveBy7X;9UZHWgW|aPRyZpMGOmD3I}7;7$`(~Icd=xOJwJH9qth}-Vp!FcH`h!u`Fuxbnarp7LNsP{s!j~9=_2LimWpP>;dnZ6lLTUrHkqI=EOWbyeBu$c=p%%Fqaa%Qg!7QBs)QgYxrr%U(qOVY*8_*D=A<}o=p#Fhxt>ZclA6h`3tDq2k-{O6>3={D)bJYQZV|_rr(rSOgExo+Cg~ZO0q}LU1}buwM&v|QKbO&5o&Y-?nKdw4U^*ZNI$H;_LL6$!%M8@ynJ-KZa8od`wj3)+z37EJ*EvYhuA8MYlcSQ)ETd~D*%J`Cmg8~EIHdL2w9+mLzF=I>Djh^pk|^k!W!G1^pYR9A31-TqF*?(EjOb9M(31hzEW94)3Gh&-aySc0%B{A9B%v{v^~}os@yrk*bd{A6GcHR2L0n~#w8xqv2oVCkkP^V$!bbL*=SZ!?ut)dEWOVA{z&Pxm67-ZLs=P4y2{zRqe=^E8xtn#oC=)DN?@H9wQSqTQM6apR^{90oN^CYmV|9sOI6;ogfDa%qh(vDW3V?v)M7%?@=CW0!Na%Jgp^(53UpLHm-qN|G5*kGG!9Ah2P3Nipvm|0&Zg6&rG_OgbTh@u{__(Z<O+&>R3$=dyj_xJ`XFezcbcG}W91)Wsgu{Gig79A_^gbvQ>a58SlcQ1Og_FJRSpH04QzCe;WRnea9n`Oxpp7p0Oj@y}S;aE>f{xvUL799(nQTF{OMCOd^Vc6B)NiqLNl3dLJG_3DB;3(Np>Ip=>By~~wvI;=;mNW|jaZ&e3uCh6lrFgxmSr%?W<|*gDcW3;#=h*Tzz2*3DF0if-HfJsN}Srn&TO(Q_s7zFCl8DBopkF5N3@gl2X`2LF3)X}%XajzFx$~XJKK?y8wKEUjT}Y!q?w#4X<;)|xRkNll8HUJlxayax3b<74qKUgTCeNE*eoouza(?|(p*dDRw5;f;j$yh6Z~IjIdC+*kSh+*4yvCq5g<E0n{gEW0prlqG|F{v5VeLP&^G8i4y8GdIEeUY*6Tr5dEek;zc2}Hu%6`d;lyB-98_I)EOv6o!Y)^w21rO7o##iA%MfZW&IXh(%SmLu+4<oKf6b|xvxM(=oSQ4M=NuQN;K}sNiG${quWD5%<;k{;tV8plFn`UL-+P;JK+8*%7y2fpmt9$V7K3%meCuyzWagLh<*bU>QXjY9ziYmH`YVV|fEaEy{v@~Bd~BZYB7FB8*h^^ip!y)sqJ7wTZPG6$5^GloOJij@ES1zY2K**5Gx&gHo5svwiG~A)r9pMWl6}>n;OiHvIIP79tXtJkQl3sBz^a^cP0`uG)7=C^aY|3SgmgmzF05FuMicc@Y`WB<$g(7ppt*}o)0L3U+V>Cz84^-j;Y}h4hdtmpSG-D<@2hH(UJ%>Dd;j$$bY4p8QD?MafsrW&1>|(H$cBYTf$)n+KJG~t9}~BmU>Zcu3TkrBQi)JL@4GO?XXoi|l9g45k43az2C@L!dXCN;h~&r$6{}770GPr%i9H^#SlWa4bT`rd6;<|lUNx<D6;heDwiCv%^kZu@v=2cH=XHpL<Q<HngeaG);2A7~wp&RySI4NW#6)yRNb<r!n6~t57!b%eCooZAIV0$A;Q<Lv_+_<^vSK27Jd3BdEQXRnwFQ19TqW{-l~7VYoTev(Zb-QnEn$*K-iQ*7m}&7F=AM<+a;!%O6X__<pTw%24)>cuZ_lXmsiz3S{TsunJ(?md0`;Jf6P_@X>7!70@U@65Gfs;na)+3(?*s{r0>gdmUzrBqq=hV6L=4Et>}6Ey_7)9`n&UBDYf+)&1Oy+<`;cu=T{+i4`@p*EW_iDMRjic3d}6kot_#+UnJIktGaK!WgmcW<%cO^r>0Yyly|%_<$cT{Ov4noeA~fI4%P^lxl7Ir~TTUNXV$yR9*k0pex}J5rV=Y96)w#|r@RJrEa~3OR71PXjbQV8_laXMHL<*4@Y=~Gm?to_$DRUxU*6EATwDc%vyd545bk>+eK!Q=E!Y3~$CdNFIz}a1xhNXxFM#!*N?#+fn7GUcR6M{9GR)l25k4I=rey&*9L?=2vKd4?`q+892>bO<@MxFS!w6U{P+?eD~?lFU<oA5>9cNHyulhLH~c0QH!<%76Rp6McvlgRTrXBEIgfO6!&7y9of{dfK)t!;~C9ueiMQ-dFuMDU`=1;XDUo^u%Z9+M1}0<p-~Vv;`v*`!5|IBi;bh~!Ufqkj;?<bkRX<;SM*GarUVZ}<1jVc)c}Uui{=+`=?1cD_J0w%j_xNgwrka3J9P0y)-1WMK#ge=Qz`275E}DrtWve`4(=5C{4%gn+lyHUGznGvn~AfD^iu+c8o+tzK~PZxJh$DRDVllo18~el97oA2<otJDjg*hP$*3+${v|XxuuRL3ySLQsi$V!W9{>2=x*?Ob6p3I|5w*$E%lbwx23*_szB%55}RN?}@eVO}5|3SWFfeRi*c!>ROJ6GBjfqC<1WK+`z+I<@;ERCP7u|n`l#wrwb&9W9?&dF$f2S!f}bSCe(bnc-*`Uc1DIx%ju{$i-i2vsTn!#g-*ku+OsZ5T<_sJF1WRN-Kg!gBM-)-P4dzN52DVO;S<_uMHjr5I2Niqq5@EyB0-NQIUbVakn=$|Tz#A7gwR8->RF_TIC7fKy)-M?^F7CCt<{5H-}60QcCA##1<bZx4V|)#2@s5c$t39(T#cwBa3#ltM20NV*VCmMs}*(urOOzJ-&&Dm#*@^U$km!yk85nTX|jo>x#-Z$Pq2aL#aLzxfxgi;KrIpu*({t^OLHN_I$vzcK6OpGQ0%!FfVFE)_1{6=m1KwQ*&uZdvIgz_?*<Sbq=0l%;?#WB1$?$3Uvw)>$Td;l)5n1Cj}oUpvrs?=r;R(6)bx-^vLr2Ak}5UUsU4BSQ&R8ZR5+G{DTm80Zk&~{Gn`(GiO=A|o12`8DQgzU2E~K})g66AOJcxcz4}$^6nbRrYLAT6G|5BF`-#aiVJ*VdwZzPgzB;Cfxz&IWr&Gz>ST7xfF`imBh+oYh>dOa_ZX=B3x)r{1^-JGsmaU3Bo#Bo>&vZM?2X{3kEO{^)S=3i?JoQ;g%`CiLI45WtLQrI-6ijUkI&+wq-n&X4&BlCme`wh{t_^WY@l(45!VsOS8mpSdHiC!Mq)k@ch&29knse9}qLcozAKuE!u%8tHn|Yxq%`R{1hH>H}jn<Y9bDK3sHhhD%@lc;LCmUABU%FT<&%OCl6?fsTU7B_F+56XyEyb|!h%gch$XrIq9C$%^1imKhZl-RS>~%^!n_M$YMtI003IEqPu?>@IZigut8VYa{hK^${Uj)6;Y+_xW5H1Z^&`IP$OCVuO)vTtI%Vs7UUtXwx^WO{o_mloRr@yrP{2hNUY%2|ysAkc^3emwc*Ont=135Y`x3yg2PNlJ|Qq~gP*&U|zn5^eSNMcPH6gZ_hb{>u+#~0rIf8_R`r7)RJQXsgpJIj`qZ`K4dlC<@S1BacrbThM^w8^(J!!9`K#k!t4p;)3QT!oLL2ZlxW5O<m<WqsEDXmn{M!QzCnB3sp9UpxwH9NNYNe;?^l93L+;Njk=rb_7GfJvQiQBw%kqnFW}702*ntH5-pF*Jc9LF#I@<#X<OIU^fEaP5ma&Z!VHIRsH4@y}{pkqv)VaD&TGDKcfC8uI1EO-6L+rE#Lqxs4qbeEI8?^gJYy$HhG1KyI9#^Ehi>;tYR2F5^;^cT2}M!{Tujr;3t#&OdUzea<uW2X@3yi%H@fLR(L^!GAV#H2Qi@H?Fp@Bhb7zudq=I|Iixx--h6mnRpl3z=8FmeNclz3d=bz<s{em`SN`0#k)?mvUxCfmctM+zNlLaAQdUZGEKMr1<BBwsnR0nqJRq4iuMps&Wc0u9``&wf0pMY0rfPR<QWc8;8r_XPUcd8u%%Q37TR1OF6rY;wCnH_TY@F~w(`mw!fxq{}Mt1`b>i83T_k%(^AL_C?@1|lWz0Z<1^<xyL%W703cA~aAsvXtl(LYf2KVbCZpo}>?-XF77n+2~YRPx!YS`GgPEfM!k8-zOX;r*viz?VSeDGga;7-NR1W!mMrLUmi{V~i!hh1S>E@at@>sq@gU^DtlMgQtmUvvM;o=LWuXcY^c@ulnc_Qwj~K$EzTQff%o_R8T84n+!MAui`rxJLNreDE6Z6Uo`v+1V66&!FYKgh~*7wJK;oCi_=mq4t=#a&|7p9tA5CM#zUi`^=Oy!EbFF7=&FB|(M|9NC-qbRE{0!6y*Mu{1fF_cgQ4%5&zA@RhdIckt8VfZSHTf(6z+`ql9N@UWzc9aW05DJdbb}+Mx)Sg>sHrKZj%|v<8m_lTs8V<`4lg=qq0jrq(sbm?Pi3`kI^v6QaNjXxm44vp@zsYpY#t{n0;EX`-;)4LjpjS4aG+f3I(fSSEHJ<EUQaN6G?2J?brWQ+1s#<ye|NLg_toa&LzN8qfYxJE(dmqO+?C*CnPbkma!$y*mp^sX8mT9NOaPMkoo&VWpn5oTUb&ER`nJxPNK99ssJ(8tMIl%8`9P27H{p;xwL-q+QCafSPz^B*^jm+boV8a{vI7XyRA4Y5YyuqEo1Dl8f9MA{@&=gzze~L(V+Tnk}VTrq+i{fLBrbb1J-3cSypv+2k-}kA}Kb@vEmF5TTMf|*o8L9Z0twK77&^XB6paKaaLLDKI<v27e_$nQr-3MAu88{Z$ia^$i7XY$@+%Tk;a+9QnHu8QYxd2u0$hzLE)7`Ej+k6wy$eYjLb?2_)dvwzUpU`t+=GS&Q2fN$Lr6kd%0M(HHLIG6+<N~O6K25V%+9`P+_zgzDs+CPex*7>S6(9jDbVC(eNB!Q^ZN8wTCiGbV=kQd(4PIFfMbLsgxQC3+rjyYT9nSsYX1bmB20&3|F1M60xTAb(W}$)3PCtVn(D*H5dlY(X&KjKg|yNdS@!0=C3hTI3%O0wK!NWkvP{L@!<0eKiMbf?O*@;1{hE9gvRLvO<N~=b#`?np4BkHrK<RLDsNB4EIRL6WhbkFajc&9f0razH6HSfWSzb7T>LT2KN^2b|3GDVS8ug^{|toK(%r-m&rLD9;t#?B`mFCx#5<h5FJWI`{eQB`W*yO_AxbDIM)r2Xc<OiO;LPf|d*$)e_#U7ic0#0@J3fA>4%(&k#wnG-_lRpE?E?OCL@=wbMR%5TKDWH{bzRtT&4nHFmX<IF>=Zsa(eOSZg&oD)aN-D9*NRvg!-*M0Gee?sa9+8t@moY6l(Z=Ep%t?aN);Qz^qpiw_d?d1;+d6o+LPP(y2?huIYV@Lk!+Phgu3s!V2IR*ytjBtays&p2lne8Xh6Mb42macux@CL>_iYsHiKG%3A-QIAAfeVszz8<;}2AI7*=)o2dX*>t2+7vRXq!<diDpZdLCBw{0~(1BCP7g*HuNNzB3<YVO_^xQ<v;j+K*3On)B5hscU)Z2MM!HvwhLCA<evpR?+mrD;U~M{qh8YMa}Ub^2EE+-nDUAY*1uIgZ9U=hX(QIyt1dUpNEb8E_j=axxyaDyB^Xy#`Otp+{5Pr=OiF1>KU+t7e4)4KWtPO2=Mt4*P_v|>Ab4}u{1x!A%M3&4rueu=8X8lbQsy}hkajN6EOf$5T@+nn1`MDxc70l1D8i24nl6mszY!V%wA7GBk0Gu@$Yb22vtHbViW5i49(!tgMZXZp<eWQONTVPd#;{v)Xo!Mf5a4yCNQp5O*9TGe!jKk&)2+A4_`PoOG>&lF7MO0*eQCFO(XzB%F)%iT%PMXRD)_JOKmzy2RuP5<4rp4Dg>y;`!I$Vmu2(EqBB(7%hgUGj;N&F$>Ne&sj&u20ieWr3*>w)9%I!T;1T(7I!pM6u}%*sGTk4cS;m~=3}gT@Rz`dhKU7^hz@m7`=r-znNpJ0=ic|L9_KuzrJ9)(P=m}NL=M71xj3#3zkScvo<v22tdd@we>y{kx+#$YvD7r=TLi7f^6fhdR7KRhyfDHyERZdGI{uZvG-3rTnnEE(g(VehMcr_xDlU8!blWR8?Q=)B?#OizbbE$l*GO~|yn4I=2Wi^_~<JH*IU--7WnqjTXI=QZLlW0!Jmiw|EeUiJ9kaxzw>4E9Egp2~lWBe%`r{>NL#67Xu<(uJsvkqz~7BIa1AH};r#C%*uY)SxOtyEbO6nBtj;K6SI^(>xVIA!C>n)yn1hD|8PNbo5tMx|nI%4*P&s;$K-5MCJqzx>ir+d>WXEj2W34Tm4xa0^>1^hfY(h3+W+i#UIHysQg774~;KZ(G@gf+@>dvNg<>NOenm%?1d?aOU~V$}Mr)S|y>qqJ$_`#KG4q6sJy>ce_FadNomRB14~KQUPn)ncV>zxhDkmb~u^LEN}4?-+w|qRHEE7M^j7$DiK&-G-M=V4nf90w8Q9Ss>X`@J)*^z@&;<*pDjnQdy2Ps2}x_GEzJEw+;}gjMsXdtkQs=kczb`p=TYMxmZ{oC%lsYr$N3MJ+w};<IBnG<6vABRx5XWPVYWCn-L+oiuu$gR)QI?hId5ap(4RBnfZhF{<v#s6%Q=x>^YIFQy&0hL#ut2jfoRZfs%zx7lPT$weHM!Y_H5Dx_%jG3<WS>IK>PLcU_8E8by~_^+v&vN+pr7QAn}k!u{f>o#LW+H|E`OMdUFc(Yr9RN|I|Rn5ubXm*2?$|4(a0xoXWsGkI_g(10Rh<!QL51#}R{^rpCbzQS0<SF~Rn`EJCH<sA!I9x+^Hy7HZBi29)7O7QnP&*btTmV?FZ-u6%nu@C1i^Vb6iUO$jh)JZj1?wU*Et_Gm!_d{NKOCFYxEbU*6f)B2;;8jq>fOC2{&6rr;tu<X%$dW*V5hl`R{TDtq}x`g*(1mNPLCOM(efRuk4%UkpzbWPlYbjQ$?-fWVl`^tIZC>jo%0@uz&h9nII?>hIEmf)CMB`Z@SzV9t2o<fJI#GD}HyghYnK}#t~Ndo9_n?Qjec(n==#;Z+^6YKHWagEN?gq^dP)}7ac1MQt#5I`mXz3DAov;CsP$K0Eai*vh|urHklc@N}=9(&k!RK9D?T*V=m6({?Y_p(35_AaIxvvO{xg}#BPmQ!iTr*t9V(j$A9KL}gz)w_!ez~3_6vC&TAWp7=ZuIVxcm8;N;E%Wo7mC)LA_KIAl-pZE+QZ2!P3ACxFF_XcnLZ;!bxbT?`>NW7ss=dr0QK%>+)~E4qEzz-sA<Mf%S25%|J2Y_Ee)vCi$gmH4Ebd#&jB$5mPZioSLQCg>Z}~WZzfWJpJN3FX&e8COWrb;nw<H&F0RU7Ryj|mOiMds;A=YVhrcK%-*h(9DDYagu#9Pc~G1ju@{pMqDTCN~k6vK0e{R4BM2yzGW=SMcg%!Zc@zlj$n(}s09f3(JZ4`(dR0hgO~mX-a}G@UAJfS&c`VKDU4!dq?Y%5G@I87`D?$??g@RVqL&)<r9V$M3WQaI9g%W<s7!w+M1Ph_W7HpfjvSyDh4gwy4kB?fC7@RTb>gXp*#VX^r$_VLzg&5FF(|O5Gpx-y{gG1xBR|)hWM4OdU2(9y(&W#|&!Y1=zWIgApX~XG`Qc9)Ev+8Q07iLt@q)ebk7LGzVfPW6_ii0za=ibq5!iy}(Q0b21wzLpYw#v*{eigM!gkA@)yZ9GZ|A&;H&Fcx!wDStm@)81WM=d#s7~Ji0);G@A{&bB3QBq`H*YeIFk2g?)e%pFRT+%(Kbx08Ny|69!b+pgz%Q^7ckRquU>o;WU{U9q3L}s^XizjcZ`AmGceRlMFXu22`xzFipjm^<mGtN`{ckUr~#16!pI6;_UjS-u~(B-g+)l=xR2Z_dZh9ikv?X4{zPi#Ab2^9DT@9*2-F1U4pR5$}&%eLj$8{(V?`3eXV5FNk0?YLanr100svN4bpUOa?GgLh+U+J&Jq`rVSiuLuRU6^gC3)oWFPYrN2mT_qsXTW$3cjbnfdE?S3lfX(|0m6eWhK-o~Q`Xj`mMQ*$nNqK9A{$7{-vd;2K8-24f(C;uKO<H%kmsIyF8XdR<YS?QL(Iy&sN0xRC3j^R7b^))3DDB0Q%DZ{Ul~S5IDgAMAfJRCst!-`?^#uq1Fc!q^R5jo4p_#gCgFWRvOV_Mi_hlmR5~>A`R!nnWVkB!m(hm0;Yb2V>&iE2eVmwUmpBm?k(7Q2hvWrriMpNM9V+R=4`!yc#8eb&~66aTjyQ$d$Kg%r%KFK)s31`u+Hk4r=lNlIL_X?P<Y^ZvrWfGHWp~g<`R@6Z0gx{;_={tvu!5o4rs#&*|$yE)E;~_vB?>IlJKRPe0xc+w*D3*K<Fr5sWwN9mFuq{fhRFW_!^pm|e#Z3vIP)>pU#tXpgV!%)d{%!d{<+Ot-!5n>vd~J+sUDu9%{J<BM8=6iR*B04xd{{Gz_;g%WVlk`Fd!Xl#EbmQw9LdgCpJ+hq0L7GZ^pWWtc^W>nu2Mm^B&N=w1M#<1UITb2DwV+;rHv(r-^^Y+zGmn|VpF?D+&dD8YZh^y&GMR2Y#t7#y{X*kJJ<2!&)8Gy>5Gh=9)We>I6QgGdZp|9o?Ap7FWW3~|SjY>T)>aWj!zWV!~=bZ?Gp|SRN6Yv~HoAMf`5AnR}l<p2Dq=G|wI0O)NB)Wbyfgc@=gp$~SUK(U3<j|yIL6QamnDO!g+_TohMftO)b-+gyctr4`%-47!`fTvy9=nJoR#}c5U3N3|`k0}sg#6~7B@57K+OQ`m_FWkf9ge}?6WgCG>QWXw#l{zGhJu=tt%VYLo~~u_@^uZE7q3DLf^#YST=lXQmAxzyT}cH?&Lu?ezx`!n1#f@0c(A$mXSY&;aCpX54AtceCbp;cKq*6~FyPz#rVUS5gzj>@3Ud_bU}F43D$S=g&1ZPznUg5t6H|`7QS=X@@$3+iw8Cob+?Lf{|H`@p3d4;&KN33g8VrHu2b6BGuCQ#=DLYt-M!NC%FFl*r_Szfl5!Xz@FR{Gs$*hFzjKZ`rxuqU3ZD2kABTt0>y%D}_Bjn^D#|?+4G*EeKb;%Plu7|0Pa$0ufx<-`}b|U{Q;3<6NE#On581F<`XE2O<!_HtN$s+*)1;uT+Zod@RyyOEtreZU?`J%qQ^g4L++(4CaAk2W|vnG{IGRcgqdre8;Wk1*?N6Wl!4uso*Uop}G^tzSMODW1RxC_V~gR-X}FT#+7>!_?F<3b0%q!qT|(Q+84@rZ4~@xte<#Dtg|@Qr!&kkN4N10p77kv~D*kD^#te7&+IfnFhJk2)vrtZ*Ha1e5;@(`Hz5IoO(@ue>M!i&CH!a%^Diza-_EKN#hi&%(%^Yud62t2iJYj#ZWz`z&=3WeZINke-a12DuC2mod4GR)TPR)j;4y6mzP8#xXFB;nn6+1?&%KSUWKMsMoPm>!cQ}dL;20zK7Xx5ZK{pk+Ie*gCo$;sR`mge~&;z)ot>lC!RTvU#paW5kWNqbCsF!D~Y=BVl37up>_37iF|uOBg28JRV#s;+(0xvLPD6xA_(;Q6%9J#4v-#2`6SUNvd(IH93@AiLkT|KKh@aJm_a7+t^fV}2o?=VPu6EwZTScm9rD<?>D`#cEh0BBh}T~+HBvNv@rE8{T;Ngkq7_OVUB~6)Jz}Jufxj9t9@7g)3pw}J>Vp2$D9ivGIPxHor5g5Zma}jKuYuaCKHdNJ7Ou|%lN>xes5cNj^q7uQ>R7vz3N%)XKJ7;@jB=YeyoQU|T||#xa=p)wZd+bA733JT;4@>4&UexNg4$k)bJ}yCO}suB&NC5MITE#tW)#-op_m3h;?b?zGt}wI<YG5rm#3UBJjzxymC0zD-J83N>INDKr6L39XyzIS+*H`&xog%K6Q~~Kx2d=o8qLv7#QGMvZGawNNz?|;SrpaETDUWf`cll2O$;oTMku{7koxikUB3OrUBCADOYh|UN`~1xHUDWMSx6{rK_hI$bOQPH!ENr^zIyK8#d+DDW=IHvpDkZiahHshNODG2fM|ozjPKOjcaW-;1Nk3`wi$JU@f&uo9wte;$$J1mRoE!=nedWwG!AS3bgrYIbv^1fNs!7KtHH(VbljbfrY@y=khbFbKpLZe%}}Hq3=-o$hh!lcUYgc98(x~Wtoi%;$M66Cb}c=XM3k0#ULO#5-;>SiPZ(p6oAij@mINu&pYMKVtDEc{UNvn#dEzzj#+p-drxcEMj$QW9Ahl%R2kCkZ+nY#(C1JvLEu^XBABBF(q_>o;Q8U&)*}NC#jV7ad$&t(Fc0S2f5Gcurw4#U3&6Q47y;al$tU0q;E8u1<cS1{3I(Y}T?%?riGhytG;PS$J&j9z@MRr?5KdmL>q1=S`G6=!8W57O->XT~HwTgvL%tNtNY)ZN-ac!~QXYA0@G_Foyaq`FUR%fH#<;csTLix-%tXZ*kd>-C9w_vd8OagJ8{CLJbESDlGYLR-eh)9-~o<u1{8)4D&lG+Z7Me?;h1Sv~5n@3MP^^)XcW3OROGHwGSU^tNMx=%SD2hU?lWVVn+U`<kMc1{8aT1o1iPihMLRMSX^PaCXYoME!28)C0aij;T?ut#h2PI~HB>6+S#7OgE(^)3dXY>PoCv<Io7vtNmfJt%WZAxX7)!ht0&d&o;2$tUv4zIdoeoD$#PU^&VejvfM|xa?<_q0Y2ap1vuUDNwWa6{(@YDL{Z|rRu{7_+ID4Vi{xx@p|Y_7;Ny@c|!b^==b+LR3YDHEwN1M-L7umn$6E<^u<~5BA3foZwo7`xzmbqvOCN1Zlwpjj1A8{VfVW*)T`*YUiI+b(7Q$@Uybj{>DS^rujgwytQgM%brLevRytK>HyzL6(+-X_?~0@G^`#F>sLgdTf3iCfj6-A|f&%MiF-Qm9VG>Hxn>}dkSuJidivI}pBbF9mFQ@`(h)-_ZEpcKI%t#*tTFx<;>==2I`A0@NWE@(s4k#tQH+ays-LGu(no&<E`!Q)bKTTcnxmIB0hCHp|S>kLtsUmXE@s^6C&zuNkBjiy84`XsW*lO%~jX^ZcTIUX7*#Ip%nx;9P+i=lzY=eb1C|_EwZ0XKjXE3%4<Kl<6?=B1jNf)nvy5bbK?uc^17j~;lnj0GBkx&bhi94Q76xdYK<ro+-F(1NF%6L-Wi$9`AhIB*=QRo-fuP@(4NaF}Ud`|0llymtSWYYrk;X$|4j=^)3GP)M7H0L+=8HKp)cU<<VHm!#gr>9#?-yeT)khDv^UXOX8d0v4xb<9vE<}^iB=?B*$>s3kMnxya169EUL0!`~yvNUhR`mW%M<P4lk&Lv|j;4@aZP)`}QF$18XpO2%i=x0S_V&2qo&&IXJlQiN27Y8HoOV5*}+jE8?w93y_r|j71ocoED+1$IuJPBRSlMq%Erb>8zR4G3}K{1blvowv)coCmVK1237!g05R$DWfr*tN;**kR``4fd_sbft31@0)pPc1tRkKJ3`B%TmKg4uhJA36fUie%wJ84Yu_tK}w6_&7fA0J}*^<QG9VHg68x;yVoNb5^Y5rUuHa!@}8tcY4|}M<3NiT661Y_BI$h8rZWIhDe14XEKbv>oK-17X79%l(l5QIUG`y(08J$DTB#oB8`4W4N)P(t|1#&!OvFuA`M#umrm1uQ34_dEb@U-MSq$J($JCHszyI%s%lX5#fQ45tQOynJCE=|<XHkt)G^?tNTAj@AJ~Fh0oF~j2DS{S`Q&y}|FB(Md_nqsi<RH6hL>~v@!y-#COqxpr^^)EFh<a3`Z|s+ON^qlAQn`A@TBWrf(H=)uOmL`vgVnWg=-yJ{#9$tUk2oP15p(|KOW|Q;#IK8T0v%MI4wXs|Etwux@M=2zz=D)eWkXYkHr1`10@L)X+i*5Gszh3|Is==#SNd^U0AULfL8vutTeb+7e@xZ=TDzBx7K-`Snr5PoOH3hFoo;YLt$!X4oB6B`ld%q4b32^L4FBTyZ~Xf={=MYiIK#SJzRTsiT)xZYyIj7@<-1(I>qBI^{JCE-9UF$-Bu^5cZ?lELW5t#}RMsILdhJdaW4{$4zP!eNPFZx-adJZAs??Q3idCaY39?@SCUzVCV)>G!9sQm+q+RV`0=V704K6%KL&xr8f33<s$&fuUx4>;@meQ9cd^>(YUzQbl;mJ$slJ<vc_cBf6$=hGNN1!Y8*7lxCmpZ-P4tv)#o*%3694|T*-#>S)mG3>yzipVjfEt;j?lumFL!Pi5>Z$JTda{3fTWS3tqahgU8t;vnkdPMNs(ZKY<yOPH^_$!}^lmNlwI11rT~q5b@0Qm3+`FZ<zVL39P#?iDw@zKg4EdO3-*rjN`T1%5Xj1ut)p%5J<p?5a8tN9!m#IyM+Gx+lF%K8@fAFCi+23J$?1;K6i~RA%E-wnk`Ntdbw!?eXEgfFR%%91sIY!nXJ8<5%)iBL|!~rqvcB-&A%xp6O4($FzC*YDyYtqnxo!KXDsxurW!=_>JVt$?Ei(2DoUkVu@P&V}|r*~xdp;95WAftLsn%b0;V(){vSg#2n9b~m)=q}L-$clUz3FA9sq+~FHM$mq00bDczZBf8(F%%H7mYTauEQ4OTk;H!#b2Z11ivPsl%8oetop$byJkBsm(}jx4HcQt1T3&3sxEyhXEdF}ZuL{4Q8zz+Si9YuPU;KB!>E*8=cwCFFTpAsQ71=4-)q_7Gf=smay{{ZGIaxeaL22q-K3A6JgY+QL!QOIwQ1`)~2_fVWRw{aMAKn#03ZBisE3_2co_}{)hBccbYm?e0x&{fFI{}E5`o@yKNy%l93PZ`y1ctgTq;F#K{7w$;w-+|YT6wWXDS4dwtIW_wcG!?qbXM*Oy>-|TCc`n21xhczQ6?_saxf1v)qhpp=HGSi|Et%v>*}v1(%A=}7yWhX$}i!lynRqYT`JplMZoJUc2DG2q%r}4S+ngHg)O%zY_~;Wt1bS$&C0ouaF-c7%^y%hI60&KeD~lch={w%8UN2V%-72xIq~CYd;WoqjC548gGD$Q6oiu_=A-jFQf@O#o!^@FQR72?kJ?9vA4)c>ui$Y1s<^>PNWW0tm{fLNlgGI+`e%w7UnXdHU;Q}hr`fu1+(qE9nVtFu&V53ET6|fT%le0$`2-e&XUgA?Cju3aO{fpZK?jJ9)gTk)te6Fs01i@1pYdi(oa0*xE<EZ_=G|eU-i-`Gz4st3bVf-!?evmp^0~LS_n#aT-A|&EY%=S8d`i+!GERx(Klpg+Jv_Tf#!q|me&^|{zkGl8YC@y4%1~s_VEOQPpw2csVY5ScEYqvc*!$$AgJ7Vt&E{GhhP;(P5iqzCH~w6QP4viD&MCaK+}}(&NvY;p0JNK{4kHlDdlodMq-u`Z`0*w>jSv?Yw4_=SeeG}cckGG{to4@HP}gfnUc0_1ZK~k~D)7^boA!;Z{ayRw#%M@#qeS0wigv!kS2gJu<KLM%V}Hjsrfk|DI(v^=niSU3GtDcEpn?5X565@kU(w<P1i*YeM5aVu1GI6d(zssP+hcGL|E5WOHyJa8wbz5m*nrZ*U&g&Tn~x<~!y@U?I+eBBY|^4kC$rONNvnSSQwx|8+>ciI8-sqDL*NOYXUqH>L<<<4kH2O(ala8`=dN&y%TC}4GZ-kG@cUp1*R<GABv)Ay$vEhDvJSjrp{$r5q-CY*`BHa&F)^A@u|`l#8a*$m?y%5kS*fF9qvs`!J}WjloM&NuOo*xM^L&-#$!xTAWkWm{df`RM7oFXDl?R?Y^ulqzh2{CC;@_6lKi}{|y;f{sWBGc%1<vPufIUt`^E9d>{UA3x-c(Ja*zCsgxhclUCjkkzDIQ5z6OR-?<>hdq(}U<s2WH^u01llt*Dd3GM5DQYU*1c?W})71es~*wTgj30o1cDu_2>4hKfiwUWBA?G=o@239VdyJ8*!x$VbU@U7QVh3@kdp}jOf>C$fa(6?3M9+lnl{Ij_ShsHh)yS7W*;(=HuNS85ckN?II7<RnlV%+bR&E9lFxPnmNtb^-DXCflt41uG8J;+6o`0o$-7)q(`nIbHg2olDFAd+NUg8m2b&au4^_X*}eb|kP6)?E4VDsk8D}kfNxB+;7V3Yjo)WTWWJ3cZ3%8RcksK^vHj<(pU%F|uNTJTr1oe9gx%y|po(>}#=`+nOvl;aZa$e4lS50^elYKVE*sm=BRVB{P9ztxdEpyBG=n{323_s*W4m5!MH1BG5Q#yzWxN>p3TZ**dCJ4BBj~OBupa-Ty~e!F0U->kYqRRH7w8jRVSkw?^j&@9Zf{m{%6TN#n(?L6+KRq%FIZmu6<)jYP@v>qB6Ufy>`a$(C)|2Ulcx%ekSdgsg#z!zeB;kQ{c>G#1Y!$8FfP_Y0OV_8*9S`hR%bi}QeY!R((>nAesxyH>2e<nZ%A;ISvd6nvwA}wiDOLJyqRE^oQzyfD9zJ`&u3p>{d9B9S$!S{l=PR1j&QN0PqB1d_;W3p3O*;eYrmw7q)mhi$*m;1Ti9>?uH**hX>v!@A(;Ilq@Y`k0pTTH*6W)x-t0#RST-NNQ>B8GorH2?l!f=iOIF89;@IoiY(gX}N}yZAjSJ%4uaoS|V7`fzWdZoqUAbz@+JbL7tMA3p>zX5f->{qNHFc^g<TM=_*Bigx`i||gfAb$|R^UC8ngRMu)R9rg5e?YvQ`E|}px$a4xK7q=;%>8fmt?Qzvsp6EAjwLpIKZGL=_%t-F~%_<BDd&}zyqfONJK1$!%YYLjzCcMqZkyP`5-!#(h*Ezv<3m}mrSAj#-f|tas;sY)M|T4XJMk9B>ERfnC$nf`E?ycV~5Jk$DBi%69O5r$zKjvEV_|+eml5B5`bH*;NLYv7hN5wLJ_&z2cyH-u48s{thjQll=KRUczSc?6;Zc92=M1vrPwPY_gIBRx|y|d{j-x#)1vASOe2mVMq4Z?`OcF=WzMdBDb%lG(HS00Hy1wx5MCm2AGX6-N1RP5bNA?QIMds8=PL<j<Wz8K&wsdlQ}H_dBfF@-Ud4~-^=eeU38p^A;I8kQjXSNB<y$N5R$1#OCnwgeRoh^c&U6GB`t+3OJU767EA7H=HIVea_djP8)$9'}

CASES = (
    (1, "decode 1", "normal", 2826763372427500438),
    (8, "verify 1", "normal", 2827019192515550467),
    (16, "decode 16", "normal", 2826848645790183780),
    (16, "decode 16", "skewed", 20261111),
    (16, "decode 16", "skewed", 20261112),
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
    return ("BASE", "STRIDE")





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
    log(f'MD_INFO v=14 MID16(STRIDE slot = MID non-WS SK_T=16, splits only >16) READ_FLUSH(judge-like: amax over 256MiB, no dirty lines) STRIDE_IS_TINY_L2H1 device={torch.cuda.get_device_name(dev)!r} reps={NREPS} flush_MiB=256 BASE={INCUMBENT_SHA256[:12]} STRIDE={SOURCE_SHA["STRIDE"][:12]}')
    log('MD_NOTE skinny A_then_B strided; per-job A credits; Bfirst=phase_attempt_before_readiness_wait; TL=separate_TUs; pass2=reverse_cases_and_variants')
    ready=queue.Queue()
    def worker():
        start=time.monotonic()
        try:
            bundles={k:build_kind(ns,ns['CUDA_SRC'],k) for k in ('MID',)}
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
        if bs not in (16,):continue
        kind={1:'TINY',8:'TINY2',16:'MID'}[bs]
        data=generate_input(bs,dist,seed,dev)
        counts=local_counts(data['topk_ids'].cpu().tolist());jobs=jobs_from_counts(counts,kind)
        name=f'R{bs}'+('' if dist=='normal' else f'_SK{seed%100}');entries.append((name,kind,data))
        log(f'MD_CASE case={name} build={kind} jobs={len(jobs)} routes={sum(counts)} '+
            'cnt='+','.join(f'{e}:{n}' for e,n in enumerate(counts) if n))
    log(f'MD_SETUP seconds={time.monotonic()-T0:.1f}')
    try:bundles,exc,secs=ready.get(timeout=max(.001,BUDGET_S-(time.monotonic()-T0)))
    except queue.Empty:log('MD_ERR build budget_not_ready');return
    if exc is not None:error('build',exc);return
    log(f'MD_BUILD kinds=1 TUs=2 seconds={secs:.1f}')
    calls={}
    for kind,bundle in bundles.items():
        for v in ('BASE','STRIDE'):
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
            continue
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
