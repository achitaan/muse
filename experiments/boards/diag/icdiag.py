# okc: problem=muse_prefill_layer
"""DIAGNOSTIC (not a real submission): cold instruction/constant-fetch cost of the attention decode kernel.
Runs inside prepare(); kernel_fn returns correctly shaped zeros (tests fail by design). Read the IC_* log lines.

Per case: CUDA graphs of the attention board best kernel_fn on two independent input sets A and B.
 COLD : flush 256 MB -> replay A (timed)
 WARMI: flush -> replay B (untimed) -> replay A (timed)   (A's data cold, kernel code/constants warm)
 WARM : replay A, replay A (timed), no flush
d_ic = cold - warmi (median, us). Event timing includes graph-launch overhead; differences matter.
"""

import base64
import hashlib
import math
import statistics
import time
import zlib

import torch

HIDDEN = 8192
BUDGET_S = 230.0
REPS = 40
CTX = 80000
INCUMBENT_SHA256 = "ada6df950cdb54aae711929b6e45f0b5cb1b365acd0bcaaac831394fc916d1e1"
INCUMBENT_B85 = (
    'c-ri}ds`bv(l`9SKShVVoSgwl*2x%yojlmaiC<%z07-T)mM9}>Bqr#@j0DJu_p`tCtLpCQ8I6Q+ocrGUdG1`hMsw;zb#--}>u+j&'
    '+1XW-c-$U@!{@_U61IZrG#pK%@u<}eJL7Kn;BV^3NivPYV7RNUe+y@~kLaUvy;|Qcsm^#fjHasBcpNlqVOZ<b8eyZh-3{99-jgTw'
    'okq3WeA2Er+Rev}#?wZH8aocDm#&(Px?!yLCqd^TREJR~93`QuSE?lyhcuq<tV1ug!@>9lA03W6!9Wei-KZCJ0*o%%RY@lfrkxAb'
    ';Wj4JTsOR?mL4V1$MDf0n2qRR67@&HbQaSKahQbhb(m0lonV4{-Ew=@>(PgUs25J7VOUWIK|hQKb9EEMLmELe9L}ZzK}Q{rj}B<Q'
    'qhuP4riluu#pHti?P7c{gTWx->uA)s1M7_kG=4Ru(M6+a825rs_;-3OyqW>TG$WepAegIhOu$Iz`SQnC`>JyhjQU|huO!nkb@t+Q'
    'ZCm~D=&c%!<6%jL6O6jbu2*?Hn@(m^)gBXUX&_;O7VrEoy9BGh_$iDN>Pgn={|iR5AfBswwc1?q*~N4^*?sir=H{jna95Ra+<!EX'
    '?vh9JrF?vPaQyB>?Z1DeUVeE0>don!5ARRZ>kr53$CCq^?!nRVhgUzo#O)Hl_3F*Z>G7KvKjMyUq*hU{!d^6r7)2`fKK-|$XSji7'
    '8W1vt!HAcJ7G{_*C~0-O@+B=@tO!q2Y{c!7Z47UBBia>Fdxj?}NC^3;V<7EZorE1}rba!)<5~YgJyqi#T}1TWxHB8lk`;|^94`#7'
    'GoH-jsDCk4<C_toDh&+lJ)NsyHoX|fgzM<jwp8@Z^dgul>bf5j?gB%n>5Auu5%viI54eMcq0B~@D$gTSL5Ev1V*w@(2DYIwy`l$7'
    'h$sUf@J{3Lpu~GeUkrHgCCmzUg(`rmj7PRD{i?c&rWewZbXrlb$1#s(GK(i;!lo3)775Zd+tBT7@PrcOMZSDIz6s+JEoV%thJi#Q'
    '`CU@eF;NOa1H5P3k%tV4SPg>_5ezIR`c7uF>t%2yb#oCi5C|W*=YU(!Lg*#}j!`>aM5E!wN-m-a+Ui9;f-6zA4jT6!S0DP^A!vat'
    'LE6-83aZp)+5?)Dm^P`=b~qwnbU<w~?K@-5qQF1Lvkm3Z_xKfW_%0UE{{Wrqs5?V@v2yXnHWA*^sG|hKq45u+B;n1#*e4r=7pV{d'
    'PpHohZBp8R!#w^G1t(%nBv+pH82*>&CUY{ROCF6ngBgQ`wkZ+LDG_=)uc!f9*<{?C-Y^;GE)@>DlHHx$lx;{}DH$r!i~6&e-=Jk0'
    'ggHX<q5T^n$l?@&(OmA)0?h`zdwTJBNDo2lY4qv#A#52XcsDSJ#&-t#vZn$ifZ_&A*@@a#aucR)HHom{#ymhh;XdI9&4q4frk<sB'
    'H0Nuf{Rvu<-A(6X6sYNZk_W)g<M?ucc-+uU9ywDjU}3sNqLG=7vq1!mdV<5COC;wyhz5KZ3!QMNSP2vusM!fLiUkh+Fw~F$La$qL'
    'C{V*G&mg9g6z3De%8b}H;?Xq1+h91M-sw|16lfF3r+9aNG6|{oTVYxJEIeL?adb^ky$%&XlWgR%0zDN&Lqox~CD1SvXa``W2qUmv'
    'yXXabgK$bH8#?3V!%n%mh&mS;vO&v7oQ1F<*AXu+Fr0v-_mK(-+T++<Q6qZc-Q8xJ!%0aT)@aIW5m1NY0dE=lFzQDm>SqxL7HN1&'
    ')p{8+QBrv@YA6EB^-6LF`m~9|VGubxIGhAA;|}1%lNp9_NSxjv8eK8~+k`*BoKY|geS-v+al_$K%AB$mybA*Zl)`Z@iv=%n2xxwx'
    '5CihN!*1$KuNIhkLz?s0jt0$TR)}%Ct2vvrt+Bv~zoR#nO`_y%7tLwX$WvmU%-Td@G>I`>3}YORpNFV733tWRe}P3wSDg>cmFJAh'
    '_<)Y{0I&<%Apx~V09nNht;aQ}4Lh$5+nVqVmW<P9;ea-0JSO5@!g{oW0b|xpj4wyb$jnB19TfIP7W`ofFu-e?q??FWA}Osr!W;_c'
    '_Cx<m$3>Jg8sPH*v8&XcbJ$tNDw!wKaF}FCKAqaL5XVl3PgwnqAOlWK;e0Ij=7PVJAu}1AaNrL35f9y&CCn*spF^grnnC@UsiJ$J'
    'g|`M6*%=!Qg?KiJI<xUCp^ZHZ;!BXnIOVlcCN31gP$)Roi$Pq7n;>T!?+NmOa`$j$qj-Dgcz8Q~vp3gZpj=>rjt+BOs|y-ZJ0uJv'
    'Ry1U)PeXTma<`|JHwWmtGbY%}aSwLd*_bJ@s#nx^VAat1OA89at*Mil9LySn3&p3*7IbNNNGGgvK&3!rXza`zGH*dlC(V}l#z{CO'
    '7^O%_By!NbiNLdrNZMswNNDc3EE5M6_aUy0=fPk)FZbe*t|DR<uSujPvR}XrXc-ipTAf1)Oy&gmy+99L@;{li>1%>0AuuHPQhJ9*'
    'E{Al&cQkY2%8OZX!qRPK$QODoU<R1_q|4QCmg^`0iT=N4tvvcLoJ_F+Nx+#Jb}NDog+(x5O=QN-T9QaW&0Pf7A#;uCpje(Cv%Uw%'
    'MmQj1E&meT8OKvuCQB<c6R!D8CPwM}018<#({nJH3?S-`M}!0nZ_qeByv`tq2=el7W{Lz)ZpMYPrLiO0-ANL}5%1j|luJf_3ZwMQ'
    'bV-Ak_#}vrN1;ARiBJ+>W90_^F#kb1Es?PGNllXwrZGD}-N_9W#hj59^`-}Ga3rsUC>n6xF3>dU%P@j|fX7S>wfewIk2!~LkH<+;'
    'X3$|Wo$(C3m|PPc1Zoi6B(rFW2@k@)92*2VGXa;U$kESLgiw@xJorQl`)*5}bfM;Ea%OFY%>7bB!qjDPGiAMuAgp(g5oXf8r;kd*'
    'oylo}&5M;xaVJO&^Vg-@1|2P&sXa)i-IP=`E9y9O^;Q)gz%ZDn6fjS#h+am9N6JyH6<i}P5ExPFZbrnAF$<g#{T(|edRA~3CSVE0'
    'E2TmggT_U-VHnD~i^^3v1ViC=-E-Ote3|o%FtCqtW0)e>&{9V<Sr9jusdi*i{I@XWfX|e?yt1c{;k4(r)3d0RQdS*u8#D=`a*Bak'
    'B&I|o;EzbHNoEirhZf<{s;G;*0tB?Q*nh|AKMtp~2}_1Eca#Y%h0wV2)68^lyG;?Lgk6}Pt|g8AC6Li>2wtp|VN4i-r|Fi|vlNv`'
    '!I&0_aF!z!dTUzY0d~jCFw^;j$pW;pE90qq=ovX7KjB3Hc72dJix*nx1OjZl|8V-|<-rDR(%UJ+ADdJ61jatoo4W%Ya#66C7NE?l'
    'mubw1z5%Ui&}9iXMaQrZ#2{e-bk&)LG#O+nBGciCl-7Z9rlDeZ7s8kkkXjjpK>`sjYk$xmrF)uKE$VpJ3^g$0Orf4aDu-#w%5m(u'
    'hd>7LvU@+PB2!T>rAat+`{@a~*uyw36~Q|&Jfx$E({e}ww0aABh*=SmJ!Rpj$+%m_ROfb`U@(Hn4i+U0NB};)5ZMout_a}H>N6W6'
    '!n{>SlDv|N&fs?PL(=<>Dcd|_Yp};n(Cy-P3<=A{(M&sL0vb~5?2nSHLlSj!sKinnlr7zEIO@)Zh9%BmkRd%Hd77nNq<jo&qf#Mo'
    '6mBxsq>+>+{6suk#7F_qDtk;JxKas*nPN6R2rH3O&1Dt_&FC4qLx<up0#BUbq6_2%R|Q#M8k2?`i`ckVFrt#Xp?WN;&X)*2t`5;|'
    'GHwx_XS9y#AQtR$nRDE7_RxegvkuJ4DT}OJl#@6Xmgg~xSK7KEvU18{CKdJLh)&^zmoU7g?m7`faBkHx{x~|9d7ejg)NPKMZK-O_'
    'l_}6`Ud1GAWZTuA-%lXa%o`8YK_O{GSmSPLd6oa(kEi&|+NbzTYL7(}#J29USc!v&hnLLg_(;NTD8?>qkjw(<&V)x1r7pqK3SoVc'
    'C3!>4_0C~AFuV;rnInP8v;|BY_JdfgXL(7-v21J;;WVsF0wU`0>)kQa*{QJhj@1i`pbdz^2&KlEU<m!5<&L163uE{%D1AvI)3_}!'
    '8)P=)F1dO!Ehfj|)hyB$GB5-Qtq@oOUSeV$#zUCwF(QIE@h%;js9qLF$bp7=L9c9fnps7Cf)pGYDa@W!)T=0AkqxHnp8A=1TY}Ks'
    'ZVx-)_FP10)&M}Xb4dlSD+^4iwyu;epr#K=I$#eYh5}<zNOs?c$~CjxK6Gz%Ol<6*s5d7Y>c#%ao0AmKKfgKs{=<)_>gWCA<Nf!i'
    'Zw^k>hhx`a;lpdS|Nfuq?VI<nh;fR<3VsWnZsN`-V*1l{)Vt~KV*PGl=ox#Wu-QJ$?J?I-a{A`<@SsF%@V@-!{p;g5@4q{EckuqS'
    'q~0AIzx<v?y8q(M;hWQcGE%*MbNc?^M4T)3Z7WCn$FzVy9_}BjqaTltKAarLX)PAg0W2jniwSicu@!|aP9l@$5s20?9#7&30sv3E'
    'M;M2f7$;Nu?C5z#JDMcKU18EGd5;o43zBgsvckF?1KP^WdSutkyeLkm__$M{dj`Dt@GuJ6(SV%^-r($3#7d2(JQ`_-ZVgyPOCu$T'
    'KcfON);dC|sjI>mh5bQ9+*~L0OV;*Z$|#~8t<;_Sz!P=@_K88%W)_hL*oS_^vBa2er|`{4*ju1@-w37695_%Gr3*NSxO1(a;H3_R'
    'K|gDc!1u;cB6XQaCLw$cGB$YHYQ!>&X#~tr>i?vhHS`>sK^i+$?y>j}fD3TWdf3hLf;Pi-X6evO?nEQKunq;zYA?K%-T`K0nEYTY'
    'NZcQfyEoAwtLMC=BYixX1kh-Lub5%@y&xLE=ZrlR2EExRWs>;tEcBLvO#>+H!bLg{6G9|lCIo=Iy3n>_jRHaUI%30cPdm8K&d|Ux'
    'jvu;%bZ0!RsQnI35x~PxY4pFJ9`w#;`S}9EL3T4Q8b?=}6%B9OxfqW{bIZEfoZ|uOiis2Kg-mmZB=RVOQ74qiOhk37spOn-F&vKI'
    'g5#>~1jGX~12t|Bw4RLlpGP1~;HkxkM6<#^($ZX%<cwJ)$b3J(fz%}etOXpy)@eGOC42vj1{oWV<><Asht<#eHb`ztaCywkF{QR4'
    'm;R?ZEC(^PJ{uBR)Dv0`av;=zVbpVA)D3$i7|A!pb$1KY|3N%tT4fk_3r%;k&*C_>3TnMK(eE(E#?q?a(xT40J=g4tJFPh&kwVuB'
    '3pWmaI*h0tPm0zD?_c3eE^;ok@9ZBP(Q9x1v5Q4ztp(BFxpwQw_@~k{9@0(5UIypIH9sn8CzGsRBW2vjw5#ID02E&2$Wp3N=tbe6'
    'n<zR~h$P5C(S~(4B%IwiJKu0cBIrr<SvfaI##C6#n~pSJQQoWZ=<n8tBD0H42VeP$wFWFEC&Z=<;GCO|MN1*h!Jn~BV=pChIs<O4'
    '5tx+=G6<q3^ywgh8BX5T8s3!h^V@<?gn!_MMbu*kc47{5V_0s7scR2ghtts{_;iDY$O?E+?G2nUS*x#h6T(mlB_qc>LBr74ovqR+'
    'Rk{Rm=K_}d6g^TC>e-zBJ6C5sej0Di&U|h^(WujPB(*G3m0VX`<>3YE-|Fv4BSr!O(U1dB8w8BF5RJ4T%*52<nZq<W8i;Y5b$UTY'
    'xoA*4n7W(cE|=d!k`vya9F%G3_Oo@Qfo0r`_T!U=90lN_ml#YU?hDI#{Xfa*8wOtk@gxj0!!VnGnI*y|n&7BEBV;5NgASEZ-gDCG'
    'aZ=eJS)3iC+{VVncg!eTzi=gf{M_j~Ai@b@>ZSTH2}iq7xvCyJ<g?THC3PMS*q6RT3`*s}gZ)9D<iY91@Il%9_uw^e45DIKt}qdk'
    'm-$xnK#A(-hJl&!i~6g-C*rx^eQ+7hlSkLVfbOcvU?$tKJs5W`=^c=+Q3$$x8N!zHfPDkia}^I0Xf{2p|1h}qeo$|JIaQk~8MYD{'
    'eCVs6d|XET;W+Ag7>@to0i!wG=zCF|OcfmqWzmnaI}tUKo+r?{pdrS_t4nFM)uWSq9Q)#d|3Iw1j7iBL0y9}Air%GjWDv}0i^_TW'
    '_R&x3Y6c-r`z*aa+ELy?^IgN&r1JrTqN0Al9O4<YFe83z$`Wq51GnFI@Aebr5%A#LuVVQ5OLJ(6z3yJ`Hy%J2;d8e~KS?XKZT;yH'
    'dc!r*4|p?nrsjsP%4I_ru37_PB6sN92L#Kf)si-MQ*#mPP%*BX{%{!R7H#(@-yISn|DDIdZ%u;f1vkx#Hn;+g2GJ-~dd;wTAz}4q'
    '(ERWm={$&|TXe)qZMqEMq6okdHojuF9gT_)Hq{4e>*eV_Bdlk!aS}wVY<T;Vx(I_VG%K`9(=uK251)8lIt1vg4RkcU8PltTSNi3s'
    'LJL7K1EA+}bJxM}zhmn~EN=)Ou&EB1RibgUXA#b57k`1fSbr}?Gr^q_AS4R~S|-|y;REX$#u%Z^LlS=@`idBaKFiy)4;DZzQ7|Ul'
    'KtfX<z%n|7p_*1{yIH1<rlxFW^g5GDKu2<?q=xWCE+<ovpGh<cu|Pg%bplvPqzlO<eZx?^K5EoyTaSlPhaUE(Y&gZ<V0nHFUOl=E'
    'p)X>sawnuwct5=5Z6?7f0-(XTPtE&#asm4fAPA>AKr>xUS~V<{|3E1VLhLK;Z=uORey5ilpel)s(Xjj&#<W8oDC^^!&>5A|<PlgL'
    'A%2_cPzMoR1QXbu2+hiRTo$EDnY&QS5=02VuTI__yn6s{OlzlUNCX<<?F^;##s+)A)G&F3T&|l+{;oh5OFS@a5Nb<kiZ4V3NAY+9'
    'bpV8`9(+OxGRnignRC159yI9*j$fY}>&H<S^S>N{w$j=$S0iSji!n`N5_RD3G-4%Ti<rrgr<X~OoYpNQc?BNGgz(b`kY+j(f~CW*'
    'hB$GUFM0Rj70fF`;cB6;9YSs7^WNvB=1s#^?}*n$B$la={%{fzqRc|9#%snaetlOlX$tsd9(pXcDxxOS86ThtVJmHkL=D2;)R0*v'
    '=bU&!%ABSF9V|?#m>2+EFaX>@r+B~IkAumDjv50;vWtEgR@g&{;eQcA)|D?AngoT}AS@4PNw9ppMH7*(xDSa}gE|s6(+Fs7ytvay'
    'Gmk^PdW;sMP6S=R0}+l*oS6i%B?9jhqiHsb>(-u_UA=ZVy$Qn+k83ahR*jhC;)*6j8zvEj9h5G_&ptRHX^r4ACf<2hx7F&6ylyZJ'
    'h{k`TcbPu1D(Zwc4R_2iX9ycb`GXGQdyg#=Bx8VZv~YzBkGBdT41x7%emW+Wni;ClnqCe013Uj<$Sd`LJE0vy!`M|Xd;Ld8z_sMj'
    '36VhMX}px+jglVzgCOp+JvS+7idok5QrZEM!gOjD>LMIW7|;&&Y%|`*r3ZW>mfTQ@k|76=m>BrNCxKifVV{K}xTq5$98Z*|Z+Si~'
    'uLwwH^Loi_*b+CK5^lFfomz=0Nz1(CvtT)D$+R?~1HLnk%G`P2^0NdM!&ll@isY>d_{Ikh!~i;<u!zV$aq4tF3A?YN&h)_peP@dG'
    'c<_LjO&r3SHoEwwbut(@@x3j58FqdC;KBC?`>$HB-n=6gZoBD53jU4H+EzER{vP&Iy&eyKXyNOV*3rRn>uCSYF||PlHGk`z)=Y@r'
    '{#3?=FazGw<J*65^5fCbhvU<OSFImfha{L#n;yY;$A9o{{~xUv`=>9zr(2I}b-q46_~FNclT+>;uQsbsx6yr@cj6`;^R!r(*DV>y'
    'qerUlTTnskatqip98TXH9<<&b{PUy$5TN`#3v(dzvHgIAje~!*4&S_cbBdX(XU|llCSNq_<#sepyWoRX(W-YYg#Zw)Op|oga-@St'
    '^z(<~w<kyYFArMp-n?%efB5;t^iq3_K3XYl>Br+6c2=^^8!TQ9p`yW~2M-QEd{;lf_myU|zWwxZwOQTS*?GF%Xrd{i>2(iiJJsA@'
    'cRYinH-QfpHjf?YB;Xmy;+6Sap^36GM3(he;v|1nW+$^ZFOZvO^f$b#YP*F;<(@p;-c?7h4l55{e)w?IdZ8i9OS=5i2lRz`cRPJ~'
    'n`5czuoyzO(jl^es35ki^S(<bMKoZM!!d^^4(MG=D-3JF$2R7Mv%G&XWnChN8wri21Ahu-DK$tdaxIqCpc{|JLr3ePxdamC2Lhs_'
    '!~OTXtByZ>c-MOWBlcyzmb?Ax;5}~FY1@km<|Wfn&r%0}m*q0v1-md^^Ce3X=_R(tRZBC}?oG2_(z}Kdq<6sSwh-B!-V#YGy~Z)x'
    'qG?WpT_TU;Ko24-xw(&wsOa9Kv1IO<8NcBZabd=Pqpd_TEFJT4ix`^#r(sD>`c@KN(fqZ2eP%9OR}uir=SJ5ojc$w%AvT%Wu%o@)'
    '0Xl=%4YkzbmLKV!NY<d=3?hzJ*R4U<N`Mr8aWYO!aB~=TjiWM<LLa*o4X*25q4AYdKFJ08=Hr5Ab_h$K>6sTjcfeBccoDYQm((BX'
    'J-o(v;ac2TK`YlwpIn-?F8F5IlN8ED1J)~3G*#X*yL#(95Qe37a{Q8Z*5Cg%`kQg&d&XZrA~A%%g@X3t8~5oxHTkGB>jsbTJyhuQ'
    'X6_xtWVWVpKomc7AD^{)liGIfeZJrB72oeob_#CiZY9$$p)BA0*ND9p_?Y+|m#vntPpy_}wMN7~;y|N|YZBH-j3R~x<}%M$zteLX'
    '^#sz!CxRHAo}RYs;S(^O;UqD>6o_Ir2CjjGI1#1(7t~60Ncc25iO5$cM|7H<R_WM>@_@Eni#r_=+^gtilY=`+SkxiC>}bAA9O%RT'
    'VB8J{Q+8g6nVG<mJchD@$gcLb#rdE#7nrDj*B<(6^HHw%m){?}d<(z0-(iaJwzkX=z9C*?h7-2l^nDi?fWQqK*}ST+NBIqr;9u%Z'
    'CBx!@APq(cpMXSqwSv~O2yq}K;seG5^_rFiE@+)87-Ou8_8{(voB_opNRE1s6U{qR&*@zA)wilv-%-1&{&;&&mIH6khOEbnEvH2%'
    '4;rM$u5=5H7(_JDF%pi(fp%x1Pi}&EA`xby=b5vLgjLHPKhtL9=wqNo=Y*9ax*s!)M2jpm7nWI+;mf6Mp`EPR5H-d{z#20aJy@4v'
    'xc8(GA`Clbf?n?}2iXxlr<V+{sv1;wt{PM;VEzwkfgry{ZS582(4giD0AAP{ac(bgRO?aDtMbjsJFR=|N1~u$H}C)!C=Jvkgi3Bi'
    'Z-{QgFyZL!?}_uJ79w`l_T$HmZ8~>6?sO9~@k>@O*@gH{+}Mu4bg(b>PY#X`zGIf#Hdfay`E;>EzjtVp)E_@40N0xU@QOE=uhNm}'
    'CdKW?+tgU2zWke)<{QtBllxO;-5gyVp>K~C?;D~e-(nSasrAgzn>*;?&j0N6KEJm;vpvi%u+jrDqJ-_9s{-_Q>25f&tRcHE^9F|c'
    'Sv!f0;e7%LIE=M`tQCqS-4&;!aF9$#0T?ZqQ7-vu@aF<Y3oE{;&NV{^wgxmqzqWLJPEPl~JK)_(u<)FfZ4g+RK33O2#BN5Y+MvmM'
    'fAnaF1r<?2@n!&6c#H(57+kg3%7|+uM#&2YNhn|R$BHd8I%rk<!fAqZLx--RhGTKe789h<A#+eTd!i-QaqlZ4B7gir%X_3A(t19}'
    '>Gbv2pf5!Cckq*>#E1NxZ+weKSb5+8zwVb7F>NDHhjjdjs2gRDu%^M$RzqjQC$N2@|7XDsd6h-gg(r3)79Wui335cFac>D>YtNyj'
    '1@`b*Vleb|tGZXW)L~(Ntr!11<xY8Ao7~f$X>>=ueZg%C(1Cn!36o)W#qXc|e3Nfp{iAhq_~8@{f~?4t7N+xV{~s?z8L+oF6W%<q'
    '0yC$)YZbZHoL11<$-#F5(>q(!`XT28uxGj`y5v~r9>8Z^6GI09rnbEao_;TVNL_DQMm2kXyV;^=mZ9f|GrGTn+nT}7-n@2hI?bm{'
    'vh#$;vuT*_#Ru2R9z--7C_wp1Zf?XN@@@U<P0|WFo!p%PU-Q7|sxz9JYxSa}zQ3aXUtX8g5z>s@wkG1}CbX!;LS;#fT6`uZqIaWf'
    '>4J$S{@GOBc7M+<-gsgHC5YUyivJ==L_bUzIV2|5b9hUrUT4C<h=qemuo+?}L3e|_40dP=%X;=*dL^a7=7|^}GUf+rB_+)h=4ma3'
    'MNfln>k5FNJ=o%HnSxO)p)IIwG-qLCXSJD^u<4Fg@7O62i@cXd@WqqB#=VLeM2?PKwZfobcofiOJBvxd-sAycWffEic3^|Jy;uA$'
    '4geudoNY&}YfPL7S7?UYP1^{GA&o3O09hMPFO0%plr&BC-3#SuJ)N{iUlP{^8yKH1`s~4T$ojm`jLbp;@-XN}ojrp9AxR1KUmA5f'
    '0t|BcBrELND>)8@lIMglqJi|ik;AZ;8}$~gF`ezLS)=aB=7W4jJJ}b)5{J`SJW`%{+i$fx*VCZIQA>1+O~)<a5WESWA1k|L(XTUU'
    'F`H7`wvDPT^kHuIN!%$hl<9{cr5;a^4CxhQL>h$NhGBUsoqk29<<nBq*~992iT<t8zc741B#!T2qYbsYyAf}A)E{k(jlqUT?c-O>'
    'zjROco=9r~0NiIbT0+BjdScA3geH@b$V8(tT(OAh_ILRqi>_LJ(qR;(v`u~r(@(h<k%+hmt=|a(ssc{G!mV8aBoF{5l~Gt}6H*aP'
    'eAp<FFoOsG=StldzMlZ&1#V;=n9H?CK=K(V_gsVKYUtmAJi&^|HJ9Vf`eT%YnXKzrHU@s9)h97(VtRx&ATQg$|7NWY!H=%$4$q}C'
    '^{{GZINI>u`_k*bMud=JGk`d+PT%c=4ic3Fd&c=;#J~WS-N@5f+l4S;Xyfi7qlH!vZUZgk;_(U>jGtFbXZ#{zGwcvyMIUfxD!Xb!'
    'B$kae7l{WzELzx^wx+j?O~Mt?ec8LqsgM~_1%Ix?ifu-t)$VR*8dTcE$G1uvw;tA9w6YYK8uIm4(A|CBL;#xBsAv|f<yt)phHHN0'
    '>keDD|9Qg}_+^K^7Q~mpdHG-$2T15X8@06(1KUIVoiz*RS$WFQhC#1Xro-G{b9%h#VD>6n7~=V(>AY~Zpxb$}{qz2t)7Hc1yY$ln'
    'V;ax7kBYRl0F|pO_Whr)u5i34-PB$5yJ)q|po^dhw#E_$ov=vUOL)<c7mh>ViRm+;9U`kj?Gy&_0N+`ihR#R9aeV{U=dFJ@wC;yo'
    'mPqEY3`*=y;m)`xqKS_~dp5X~gI*sy<!_9(#j9<XK>!Fk%m-?BEo31iSk<<bIW*amH#GUbf{}&z-OxezEs%3hj>+X*i{@DF0Xejt'
    '0!m1;K3pwS>46NMB*xp_U14F|llNgAzZwGEyO0tm^8$O}vKDJB3&9oalNJDIw&Se!%evVa9~4LB1|uE#HNBL;WB*<(Bzj)K!~~;x'
    '>uyHoE-tNF%*>d^Q&5<FYrS{hpDtPf*p%qO?9;U$)QEj(wJ;j<){S;biP<%YKx9biVn-kjumcEVl5IIr4c5Cg*8nTFJ8%3E7)lj&'
    '1&>FI3=zFq;RSqH&7W>jS7Xj7i!JKYdIJ!mvEC5m-7h?dn?7d7-QV;J5hpWyDC?Y_8)}O`Ti7TmKdKESC2h}Ghco37>9IsTN3-kw'
    'eBv(~U9FU7Qcv^aQX)!cFwfHiUO6mLbA-2bPk!C%!O7KpPXo={pjRp!=yF@>O+Qvh_BYG5fw4UAZFq7J(vMqT#-xSk`JF6~-S?At'
    'I8CWQP)eV)o!?0~(8Ej#Mmkco!P1<JPYCJ5h8`z2K97@r1cpu;8cFSDM&3<N^P8~>o5P0kzJK%G_k09Jk}&C(y#RZ)?b|v@sngp4'
    'kz&RjngiSt&Pyo-^>zLA5e|5EuA%MhRlcu5M>SR_-@jfDMxE<v{UQa@?FvwUbNbi6roue6Y0%R>xicJJGl+gCPI3Cl>^gctjaTAP'
    'g{M=&gl`3o=EF~mP`texW>5qbgG`?BSx>j{8zA%7hs2ix1f4gnzXT9#p*@L{tqZdU2ji07e;1LSQ>z8A)YXN0j#lt6sq?K?Jo4}X'
    'z3O(piR2Q*br8oqZFPcV`pk5;`HlDWMS9ZJf5F%FUoIlaJ)#}YXx{9dwX4$svtO(edx7WPAGd#~XE4g(blX*79FxfK4Ceas4k0y;'
    'vv)tf{*jP<JuncBSp@_=H3KXVt-%-}!t{e)O6PPGji5Y~xC}R#B1!8>VDs^X6RN|Wyp=WRC7ecD42>Mw)I)4pO?{pJdV0<3<~COC'
    'b`1rJw9ZbhG&*)xDelo3(XGE0;_bBHHP`}b7dPqYw|q7|(0-6PK;_w+?0C1dVOsBhYMtz#zJBxm&FO)@=J=j9P9Bj9he6BEX;XC^'
    'A|F)Ui<)~;cP|>HwG7RYb-<n`ZsYA*5qDr(-Jmh@Ib6}gNM%^t9_?J#wkz@YrqUS?a2SG6$p1l1h)~eML+8%J2GNw?A2v%5AD13('
    'mmWUh%bn80r}V=z^&6QFGW<SKmMuC_J7?8%98&VT=KQWZzZ>Vi^BoSjsy^9*n)y*TKN{xnGw%=>?7H2FJxxVN9I;GVX>JW;(3>x('
    'LFvm)B14lq`8d2;M^D?$&+Vw5cEq&dC-tJ-ivw*{gpPw~06QKhQ<xoL-kz$<w({8Oec9fsPvJVhMHslXL;rtD;9?i+i%XQx7$$0`'
    'U#aZ$x1Ln)#&^h@Em-w}ev+TjPr^uEw&gFH$KSsE;Xzm$4g_8w?K8N)nc};(T&fNZmWxZZshY(1dVTfHHwq{NU$f`9{u<VH<7|<B'
    'cUh&!1gtGKAJZ6}=dyU$ctck3Cwh%O5TiA^@u=Hhlt=fzR!>MMefA6v3|r{cUu>wBX$W6!$)Ix0*(d8XS}$mHXKAB5>owXaX!PmQ'
    'MxQQiG-Yl0@|K!q2Kl;LA?kZw!@t$r$F*ja5WHO_8bNO`Q*<jgYUaj;oC-_Zc+%Le%O~}G8;zB1%(;zub{x%o8;{p*BOS+fzKtg<'
    '+Nkqf8d-SM^6;pwgh!?gGmcsw9<`P5sPkMJS$Nd)@TjeXN2ZN*9NYOe^6)SO4pEUuKZ(b*Rn)-$38B?99Y_TA8%(uY3`wLs06PV}'
    't!;BV;4VwhHNt@3=b#jhJe*w26I5*xqFLX>!!7lDwNxwZl%9UlDZ<!i)#^bJ`UOYeLDZRY;%g$qVo6~a?EryyM>?&}V4fz0N-k!-'
    '9xBb;kFvIUJNGB9_avuTTKwgx>)$3goB}l)p-vQw;#|}SgNB83kN!M^qvO`rh3=@BAk(q7N$+MU^B1Jee-U;5i|F%TM4??Y=u-2v'
    '%<5Z00ph>f)xy<u*7KvR=SQ=iAMJX&qZKZfUj0ZdgIZGCLcH7NW=(DqE#F)6eaF07msgqCb<NF&+-xlWKCIRZfeK2{#c1+I!dJav'
    'o)GumO$pzUmd1-MF_EL)u-KB2KhqKc_^e55^pRG8NjtwwL*r3!&&^u3`2AYp`%ktTOWLp7_s4jDd{_I8!uQ?wKjF`(`^ZH*lk2+$'
    'qx$-U>q|;MB=NY?52Q%>XSMwEm@l7{9`5|t%0JavUj8W*dyLr*asMh+&a>g2bF+#GKImMv;&4#uPiD${cku4T{&8jh@UZgT(T~1~'
    'K7yXB)e==KfAz1w%XuhGB%RFHCol0a;nBhIF<sM-moHBbD{p?HX8m0~s3ZycVHw`4oGAs0ix^o#5Z0vq0BF!OI3$x{>%`rNMuQN)'
    '1J7!joH2WATCd}<+oFXx*cuqs&26FDJvGo&f^L^F#@c0B#XZofSym<A-<FQiTsId7V}8udkB{cZ%{t?}ZaPE#Jyk5z|C^r*1}=Z+'
    '`tL{m_oiS%>rezbt%j}JX3JHNce`=Ak-t`PHFH(VT-7sIjlu~Jx{`+^x8Zk2JC)mJg$sii0~>Uxxov*-u(`}uDT>mVw;cbf*<W@0'
    't6|mhG%rV+*F!3=o!cFSK6!vPVSXB&v0$t8n3-kvwQnu&@Zdq~w3T-G842%065fYfPbAKUx?{1QHm^>}GThVlJ~ibO1_?JZebg*z'
    'J89FKK5Y7xiu~Pb*jygcz#8VdWXemsY+*C&npxJ&sxDj9WV&imo#jjsn^&|1tac1yyedL6V?#S*qgILyS*1Y~E&n3E8}};s!*cAC'
    'IGRl2v1C0H4_BKhAW>d73i_jQf`|y6a0Vo9SCwCVc>A(-bo}A<^O~>ytxOmi?zW(v%-wpofBe?F^mBLNQ=hx_>c@9Si{4<Vk@nrY'
    'eVZR)Jy$RB@`534F2?Uw&R+bQwr)F)7oY9s#n~FN1V7s?zfjR<l!GvTvd7c`a9NkG1K@f+-`_D%#$tQ}myIO#4a|&V!BOyY4yL|_'
    'Js&$nhZoJ^8;0eHed;J<@FpC5TOJ0#A;)E{z9Swo$xZa}BeHd6GkIw0Wl{w~<k;cx9-dgDYSnsE6R-f`7I$0R;QGStwUQeselygn'
    'z;b~1eUHfoNl(?=#%ub|hv-Wp)E^{T*_R~O!glVi$3v9Yk@2Bk^p%JQ0-IN7+s)hU=DG5={O4*PR}Et0J+P&v^r(Q8_d-AX<<vaR'
    '9v&&!{B^NvxZNCWH;FHHEGnCjV*XNt#sYtDS4lfubAUp3C5NALuVeRH*C~1eCg+K9p0S_FX2H}odCD9w1{zuEAjk5nEE7fle@y>}'
    'H#K$laEJc?6z}khc(aCAYk0SYmurt*8R8+kVb}0w4WHKVZ5`j%xfgs3sqA4L-`3GnogKIFMIB$%@kIk)G-^5Vq``y4*A0B#fE@F%'
    'L7x+FsP)={^Pt|uT}N)|I&w?bkz2ZuTkP+q^ZT*$d)xW_<lK#U(>HgV7oNJEW=7x+q2`XE<_@Cfj-vLssJogwpxP7n!W~J?9ZKCD'
    'N<9M&cPRBnQHOPROm!DDb$2LrcPMptD0O!z4R<Jw+Pw{D#OY&~8OIU5&DJGGOR3F&1#N02Nsc@T(Foj&_7|of&HtC_M@l~amt6ec'
    'M=n4L0z(tpODTf2#c&D+eqNS{CYuYhEOn(io!M|kyKEfm44e~GDjy?5ENwh^k9or&xkTs`XAezG>8T#)3BHI#(WEo8C1dTCmV_KO'
    '#idClk7-)@S=VdJ^bJiG$|ms$Oi&B*RB)cbmrovJ|8S{H?k-84Y$SA1`Y_eZbTI(mWn--u9D@WgfZ%8$fS2g%h2d=t>x%4~ESf*A'
    ')wKusy|~ux=3mdWVra{G)_Nh&`h*UKtvfl^C-v;9U|^Uz9gyCDaEhqzmTG2J^FC+AlX?LktAB;1oJN0BWeyG;ig%qB_c#Q2DF0Ir'
    '0sK?IIA$2jZOvUjY~bq#{DK}ft8z^I?@950!AX(tFmzTfbXG2ORxT_sW$4fNXD$6%Pk#!a4QJ{oPzU32?A@wIsw){us&LZ0h1n5#'
    'l|X=NxYIRv+&B+WkaLaLcnq^Y=kKLsCze{&ib&MN-{p?}y~EazSVHd;*A#-L{#~z1)eg?{)(=rS@0!yd>T?8I3)beM4%es?mZ2Qf'
    '*#W)IctFEPlB<=Z3|woE+HFN_W4#qV)`>C}C-w+lp_ysHQ?+vZ@`#<W-y9qqaJH*5N46l1#jZ|yVka)~?Lj*k$8E_`%?_}BE{}tX'
    'yW+Hq$<m4I)ha!>O2QB=4wdo9>uT3wjP)w@@vYN!EB@%^-oC|&5O@8YGghpys%2_mQ~N$&;au@zWKKaIQv7_Tsu@S&D~l#Nk+jWO'
    'qZ8XAuk@~}RvyFZO8-uS(FG$D9d5RI0-Zr!R3qT0%4=lsU=Kjj-ogK_|5C2OW{KDro5hhGaiu>_x&%K~VludxQ+PX}o_j&!V&NK2'
    'vzX2@@7(NK&_C2e3*oOluYBt|60b^Yi5x+#r&T&~ZMdXn);b~Q^+9?EN!H1MJyN*=SxJ1FI@09m<dn$BPj@+JYLELJ=(^6Fm7AMk'
    '=R}(xnt;IRp(O!zMp4ei9;m;HdI1L?Ne*3x0$s;*a_pVVU=6pLNq4Y9VTz#<Sh(2Nc)*&|z0cATqZ=7h%Yr%a(oKudGQ>pA)ijR!'
    'ARTEIUqfCvkq(2PT$b|ScENOVHk}dJ#$*wwFazC3cuU~j(<R<OXIh+ioKx#v0BHM<UNCV8d%ckB8Y!;tD0Sye+y0$UT9{`s$Y?}5'
    'YlhC3!6V~LCE0}6HXD<q-C%^{1jw8QS37}$FZVfrw@$U811h6Xgaw_-J5P1Bu?&QnxO^m7q+P?rCc14QpUjGd&3Rx+dL829^R8L#'
    'AhArP<n;L(a<btlW2B>z(CtAc6$VlYPAZ&+#J6tnupz;HCr#J^$$_};No5eX;k%PQNgH$tqHY48ZPHgZD-7MUAJ`T8m7RG)BD~Y6'
    'yQe}TukZ0^3#skC_kR@%8E~8`;X)aSZ-BxRlWq~=&~xn`gg948%^nThWEh??P4eoPs-7tk5c^nK=M}zps!?w0<Qbu4``Sa^D^B_@'
    'w`l~((zIPI*Q!<3nRf<UBYS|j&75h)M7Vul(xr%AuVHMN(mhX{>u6rDzy4bFc;Qw!xP?{mh@!s&$EpG3qN5vNj^_+`c?g8?-m8c5'
    '1!ofb;}2OA;(NSMmibf0t}LN69FD}*6(L{9a(lN`vnx!95)v{7$b}C=zGg@ZB8Qqia}ZUnGeXV&s@q=;CxWTA-I5@fizp3a(m_cy'
    '2o~H!9hj6x<JFyryo(Kl$C!}3s_{#xX;$O28bmCPZm@0|Y-Gcp()~3UTj`gtFCr~7XHEMYN`zQ32?!&?hH@l;IUkLF;o{h+F?r`6'
    'VOFvtsV3l}>Y8FM>Wys3RVxl|AmO$JXcai>dNw~%axsoc>ZdkXUC^WdW<Kq-%6!>taQ)mCfVG!2R&Qh)YcFYxUUM4Dx!DoypDP=<'
    'gPG7g>N<AkwhrIBmC-v34btMRY_NT2g9blnXzBETAvwHQD`NF(7Ia&ve<YXJdeN?FkUhHH^RCf<a?u;KZpRW0?Ma{9Z%NT}vlH(1'
    'Q9X<8*K3Aw@pzNjaACY#&(|2|-rDchKWlX3&h)=<aBIEWESgB_hao;WWiB`5l%LVLcR2~1q!*JSP9dwoSUW1Btxgc2Z7R&Whsl&>'
    'jyv5rm>C%mVHjO{IkqJqf*S7?BySmCS7wI3KGb7xb%?>I2Gc0zI7e2I8Y;eXJ@wf7QFqDEOmhsmH{ILHM`Clpp}vZSL}<2@eK3cV'
    '>6u56_$HN3;=8Yyn`<V58pu)?v3lbEvn)!VvyhlP8af+^$$aFviR9npWH8zv@=kXWQIrJJCPWK9T7>UbztZ1drQ9;f9lv6geuYrL'
    'St<#Kq?m;AejPwE{S~|*BnzED7zs}NySHEy&>3`_i@?dGgyb#+U>HZqWa85y!TL$qC$U@K%3~xv$!#tWZchB!&FiINoDC+yLPC>S'
    'CdS^y++2J;hN`hyPxgAR$op`i2kmlM<Y;KK4qR8*)lz=W)1M#HpEqVe@#hh5UYgqf6HBx1$`}R_xmBFCSPkek618*LqTiidtLpOX'
    'Wv~B?UVZ(|?yk<cyXhvwJ$F;Hs_XS%XYms0c+Pgtb$U;vd6VR#km=J@y56?&mP(WH(@5Jmbfg#@7edJ-E%Gz{OwdpBJf9|AIzx)i'
    'G8eId(vjp5N%5pS8jnr5Iq7%u={BA3IOm}xN7RLBQyOHlNttB2rp0c%KibQ-*bUpWzV@%wWnD=AFAq^WlSNkZ?RC_>-7Co5dlRBc'
    '3)cg)NqpV0V1AOr{Nx>>x5<UYcvw<p1c%H<5lZAT9s-m>;??Rqz5s!`B`}`tR9AqK3$u45iRo-YvNzZ0lL~;jMbZgxvug#RJjIDz'
    '2Z9i0f#Ba@o{K}V&SfLzb)(p1?lQ@qvOUpun9ilEWj*P%LtRn48+dWfTa%Co?x@*zucy7U`nk!P`qM9`0(a;jr_cb*wW?D3OBUPC'
    '<<HKy9u~FEUec(~48<eovHsWJ5sJu0zT{rxzk&+W=Cw}qjC1}o@y4d{UonWVxZX+S^7=~3N;25GitdbvJvpgdk8DC-)a5oK*OdZ#'
    '{oUQy^<7c5&PMzS_(UiZrJxp1#WXRxodW%A8*UTq^=(~)|HV}>Z*{MxNl#aN5~e{D3CWOjdxl+6M7`jbd~TAuLms`dUh1~yCUJB<'
    '7%v^w@L&${>hSsV9c{Znk@C^hdJ#wxjGsU)SO%}FTUF=O<NS}WJLC-W)Pw`=45DVV+0aF$h1l4;o9B%Vbg2vC-r>P2ajGSo1KntP'
    'GXRnm)oV~<R3l7!QkVIIDsu$>mfEop8DkR~WjwTaggSK8^0YW))ssM2&SRoG#j8iw;7-Hqa(ZfGV%P2o)*)e60Yc5~>W&-=8A?^H'
    'H~sYRy1qJdQuH6vVsc#OHsZ$W$YH<JY}D=A3aCxNQI6v^+Ff2`Cg#wBm}<BVhB7o52s~B5rIh*$JEQlP5BF*((Zk)raCd4?d5qiV'
    'ItBSsJ}IAhJ(29I`uJZ+&yaon8xMPTPa@<pj|hIr^z4uWX8|-rlCcLafv`+ve8IPC=LW-pF$?vR7u)SGHoKT#ydP+96q}$?pJCcE'
    'Cb_u1XV2B)W^MbCK;O^sE+g&*L8X@HpdVDnWn6T3FfmiV5oK{I0G#t;XSf@;!IWw1Bz)Uok~V;i3T{S7D1X*SX`l><i)}_+!ZCD<'
    'ZjwYT)Ra-uB--=c1nC3K)nwY!snJU&xqT*UJb`)KIXlci#AX`L!UWKk64-`L5|h!lrqToj!1r<$W5P`BI+G8Q8p5X~IX%Rd<QZb='
    'STbz|QhR93hG!_X!9@_Z6heDz=gtWFXZb=L{N}E%bKvv)XN~j|8ge>=9`fAxI^o`8<(m}W5WO$;kEKjd6)I77vLa&VyH$v~^HDHF'
    '$&icL=rU2enPZ;WFI|FjK$OlneVKECKQdKz2?J)K<fccS*TktqWNR{{wfw}0N#iWBMUH=f*zcKS`@-4bRQGMAXLG7s2G;;nSBS_1'
    'ZUV^Bfx+Xnce_{oO>JA=Z|3;>?p4oyo_`qT9!lDHoazoKDAAM&+Cgz00p*EExVwu=1BmS3h21Q(f?SUPx?3h0oCyvf5G|B=>lu5V'
    'QyJQjBNUtzt`cXN1|D;jTtt?eJAU!#_>kAmmJ6t;4{jN;(OBFCyHdPVQvM=KL^PE}hiJn!P^U`QAJew$vKfdeOKhbe&>qyLgl#kG'
    '&)}oTjvB80U-iOi=fbZPNPjRWBFiPo0^cmJn8q@q(UY;=a#0jNdH1zCDLM9`mzgV($kXfNqnG!KvT&Lj-&EbE+7s&=cl;v#dFW<='
    '$W|KDEM6}dBq6bR?@x1n9Mh=FN+eJ!*0PDU08pT{#iMu@?&S?)1R&_K#q}lC(3J=h{^HAE#GdmvW%CVP(B`TTd1u3?uJz%X=Ug)<'
    'L6DglYwcV}UZ8<R9|2^ryoTZ$X#?m@gl-4v2__XOV#4NISeJdTJ<%g!+v|iClA48L1FbKd6u11fsy)VaGnaV4==7+EWoe@1739S`'
    'esPA%9?kP~j5?T@23CUg2mj3R&qw@olL_u$0V)SdCRUo~3k}bm(**2@1|1)sg?3<zVcK#xpa0X?MH#11EGb<jxID0<uG!tDs)#0Q'
    'LUL$Bvydu$H6N3lBv@-d6Mu2hvHzd5w{Fzg?{;OT9eS{(c&aZ?FfZPbwG_}N6`aqobn*Ak86pEh&sq-1#gmksA&ofmrrXYxi=p;8'
    'D84?DFOYzj?{v*5^W5k}&*;(llB6M-QFg>k-hpNN<vXMf>13jAF>ySU&3t{<IcMA5pxfztckO|@w&7ZP(DxUohaET!E<@2~HoYob'
    'uQLDmpF4MVS@N_0Jxk_cz@@bWOo*<sq1If^arx0+e#GUCy}a4WHnd1gzAy;a0#`boPw9p8bHykfVu#&o8~b$UrYPPII#nf>r_$S7'
    'y>fiI6fB)lrQewIV39r+62}ZA0x>T#whwNyP}=0CvjcR%#U!>E8d)N0NZMJJpq=v+Q)*L0s6IJ*aiUI+Vbf(fEvr>g2euHDCeVgn'
    'Uj&KO=z))qLzcC=)-vTCz1US}5Khm}AWe>;bogZ`-j_#*=9wDEeP1|9<7Ai#EU54R#P}s#D9dVRxUmUtO0g+ntLEd%w~5oaE6c}f'
    '<13qIwU#K1;J4(|xT4FUb~4tVIlMM|gxatM^S0rqxif9qwfr=Ei$kNGD(0$1TNEXlJx>|v`?0KrCQsp7;a!tltiiUn-3uh(y?-!p'
    '`u(T{B||BbiHt<2&*DrM%O2)lwzUKEA1p00csI}Mvhz$5AT0yWn0FNm{3P!Njbqy~!a^!iCO8Y+gd0U^GkDVEUZAc0T(v=Z_{}Z;'
    'YczEyl}nbHPAb<@yRG{h;Tk;pj$X*N+Qg3ggSMGm5^GIDp(Ia+(hbj!=~Wu7ev`yET%9{b{Ekp|v6zGt1*b9iZplCD>N!5ya`lYv'
    '3&&`V-rC|3WS-#DV|*%1kNz%K)$-uGHy_^9i>?||x25?>-j;0hF(m+#P8^W<XB_Q_B<YP$<Dm0mOM^<h=Cy;s#4?6C2X~PB=Yjy6'
    'd=f3@(Kl+C-;L>DNJL<o76gPhWhBM2x!$0-P*2!=!!EC!SSC;lQY>qzIGm2;W6p3mnQC9GU5BOi*k2X)5Ff?0he!(oU}-K$%zIE%'
    'Gx~tcf%**CXBl-8rTFj(I~yc>_=U;A<uj7fFZ2p>9Xeg25p*R(7oMJd71$aw8)Co`bZF8ks?(dX9^J&+c*O1l_fB>omHdpb4Gdt>'
    'tK>z<(+{Uu49z4z;bU+?&Kz{G#TqqY<6EuH;G-w|Fw5nsn&#J~RTMX+w$l?DW=TvbK-M0qJHkqt(U_^cyq6QeF10J^C0t3VIsUaJ'
    '1c(_IX&36WvbHxmez_R|6`Xl}IbUwOoJo%}lg@9pL|g#z;<SP@9!tiBHTxznD!mNqGyv(l;;XL}fzcI=a^lE<-r`RmDMYfdZBE}Q'
    '<0+QQ`SSW~c788r6RgmsfVF`2=I$gkQn#!<TWD8f^Iw&Cj{pGiz_Pyl6)~eFO=Yrd8yB}gYwR(p-7(PKA-89nEBgAA!^-`s6|}EO'
    '+RoY@|9A$pmtW6>@VEolW4_bhjbAIdc_p3i=j(kxU-tdS^}heO?E9PbzQ0-c{?q>i^mnis)uylVYLWuHzC)_=q?2up*1i?80H#H}'
    'CZQu9oa@I<3eGI*X0Q26T$unFea`L5xdF1R6aCOp(-d}^Lu1<GF7oRJMDC2=N}{#*LSI*(j(*l(e?4u9qZyKfw>eeGDm9*b%ev2j'
    '=vnMb&r@0JUMhxESRJI&YVnC=XL_ZNldG!U@K=1AG#xy#XGQZotKd`veQ$m0(_ExJ#k7?75%0Sq(l^i_G336@(@p(tZZT{-YBwV%'
    'fBLNPr}~yzR^Kwq>RV=6eG90`9+ro$coKWLEpTtW26pg(buEzbd8!PVz{G?I2Vs9WtSAz2>(b5a?nT^wmy-z+&%};Jj-5wK-v1gc'
    ')7VSUk}(vS1(ugGCXQ!^-=Oo85?eAvTgq~_*dJhZfVtuL&`)QOS9rowK1@(to8Y0tJ3Uyned{2>5-&{rwyGIR+u&ifZw0(sTo%g>'
    'ZM~LS*wDMe5L`Q7Fd4LkL?HGQf4(MO5Ots1S%T5F6k{b@MnqV1uo633pnK-9Af&%`y34-Bs{)bWd6sMy7%y;=;LUIC?!cxXdAi9T'
    'gADE6Ve^Ow1#kQ&&$-Exy13An1?_190w&%JMIRU0-RZs^jS$el-+Y6Y=o=^E^;#R|@!VhDiV#!j@jiAFYH?jgQYT}T?D)TDYutcr'
    'Y+@!vkMd9EOP_pP`s7AG`IPHDfp{`Up@2>KvqgVua#fS7x?I)esv%d6l*UsNcd<uFlCgBG`9=NSUo`Li#g1QyLNUluoLt4zo<}Wh'
    'Dg;UsKegSY1%${SmLK<uQhe8m?FepG*xF1<mak7L@WEkDiO>E{GiP{{_cLxg^-a~U7imNa-=$74n6yP1;P^UoLsFQ?mR(gcYkckB'
    'eS}|Xq0<NumjxoA1oo@`lx;G0xDwr8_xkm-%kxFfR--G}5DskbPucA~sW~_1?q^wGsJQM{4zu@>O<>M`hnGNX)Y73&N=o08#OdaK'
    'hT$IHoUyulb)m7?LE?-mohXaH<Xf)TQ-KEpQE_cPVWt0;`i(<UCYA567;QM5mlaV^&C$ihg`dSIumE<!Kc&amr&R0y-7+sPqbC-n'
    'aQPd>D62f<1ywTPp~w!Q<HR|Msj%Mu7gHXy3swu~_U9;&{}Z*ng(_N(E-D!%U#i}9RDD^=!%-GHNBPbw0fdiqKFb>=Da(Ol{K<me'
    'DtCsL);O}$TDEDe-9;*Qfe~hs!s!1d6{LVD#`sBP&~53*jQmp6gEiNw@U(0raF-<;DC(g82PE_|=6*t)w?Ao&F<xS4oOXhJy8N7b'
    'fbmP4j1%_qhp_~DapF2Sb4AC=P@tWa#5q@MNO#J_T{6GSE$ddxJbo~@?DeO9GqtuZ#6^i+4`B1DsF%`cBvZ*qyP~>8su{VTYB8zV'
    'eh7n;FA*L_J?zNFNhCQnh!CvM4~t!^P)AI}ZA8<%3DUTJ1;<K$0eKnzUBA^;71JCk1@2L6Pzd7TS_0f<!AZtL$zeX-3&7PS7qr-)'
    ';TC0BY7@xreLeTjcFy(X*P7S--Iaf`T6$TE$y}0GfQm9OlTcDt@FVdoNB!h5F7q9m-oQ&pJYbCMgrj6;d{XQ=^ZUIVj_kH$4UV&u'
    '03KTdZGm6D4c2$Epk94{Ni*&HHN(Ypilv55+EUJ#zA#1GfH1+&X*Qs$!UXfxd`~mMeD!fpcwxD^aXxR#&9;30*j(}F+vbWte`2oE'
    '&sk`J%-^xMR!I6h{q+^so8BEwaap4E+hbG)N;ob7gR^1)pJ%)dG$SHT09o<f?2R3}H#nZkkWiCNsBEbARck<GatCRBMzmS9Ll^B1'
    '>;`MXk=h3{gc6W2o!=b=`x=HUiJxHd)hydBXtFkq$Cq5-H%+p0Sa<qJ{le__V2W_tMzm8-Kh}|ZcQDtE|H#gw^GGHZbLj?`)!9iM'
    'N!H!!3KO`GA60gXD@^>ppJ%tW0-W3?(mj3GtnQ)2TMgutI%xl?lt5??s+2}(AF7m0<h`g;T45V|?A*vu4BOZf=Z4!@N;&9B-udjW'
    'L5MF$nZ7)2T1uVnnD2ScJL=O4f>oq4Pw_}A%sW-<?$ybu%u~Y96vAK?=awuiRhzG+G<V!yAW*Qr+s=%-y_eaO#&@sxY4o!^G&RYE'
    'nW@l0@jf&x>koG?GB*2jWNh~5$=GZK8CyBA0>u1l(BrP9awy>}PYGvvN;u0?!r4+vILlMQS)LNk@|1A4loHPFPYFL4t^NlS!&OsW'
    'MGqZ8`jE!R-cH7KP=rszq_Oi2=YVuY;`uJ*H;6RxkUuIdv(}JMIvvNM_-)2vqO(KV)+*}#7?=#vu)DiLc7+HXxvAnfG$-t`BwOU|'
    'D7vCGBvG1pYt*T&6hjwW8w+Jr*Rr&(Fq*d0jpqyHSBrfMk!CZmCqR;yAVEx&n8ev`p5bd5nxfy+er!mMsN>F&i+#^gMbu4e#DWJV'
    'd|NnbW~IHmv<yo$47--?hYN61_u?-q<Bs-EPFQ=VJ@$yJe}78<?)U|oyYG(R9VlOX1LfVf5}~}{H;H|u#Ju)o?+!j-qWORc_yB2~'
    'FNi>WcT~U%e0L<g0R7D~1L-X(Ko8}<zPauh@ZHh!Lm)=_TyMh10IQibq>L>QIzxY%F)z+22IHlsyf3GF_*3WjpFYKxp5b?>7kJHP'
    '=XV&0B#7cK(uW#II{x_o+353y5eH}CS1FnNAAEeTp=oifP6Z}%M&eaAa=ef?k{h%2PJ!84bAwCF(3&+|`Sjz;r^bxE<Xp^s(q1{3'
    '_R7JuR}QA_4yMqiUgT9(7+6#l+YKulLSNgh=Lo~u<Ug)O`?*b2m?U4&aQ)|P%s`WjRk`_>nwcHAH8tEm!^|B1cbJ*O0yA^C+RPmO'
    '2{W@0y5f(UU&PLAkCO#U^QUZ4X~VK{o(8agoEL!evxhfa-)6#r62#F3Y*Kbg_jH00>ib-bk+lIHn{6<pl9oFg{iJOtT(tm&XjvaD'
    '>2;Pc_c{mL(dn_t?tFJJKG9zuY-<#GC+{}ZiB94e>8fE<^OvoMk|*OzeuSfL$#CTwZP|v6H9nc-qk<R_IraUPigYHV+&i4<7MWnD'
    'Rpe>7@=KDdb{LK57JY|r#Dr-*@aUUo3b~AbUlq5F#I$9F9Lj}O<qT6i26duljUy$B#C42$xx}Rz@J^YgRKNrz?AYpc)GsK}KwtP='
    'KgjXtS&qVpMA{-CWwYGDY^@W;;OULO2Ft}6v%pgEKmp3Mg$|#v>y797i;d@c&GCFnO&TV9Q9+sA)Y78EFEtw6k(}^w|5Pyc_yZ6L'
    'zutX#bpRT?S}^9%wJE*8b7L3|Inx)l|4rVBqm`GH?G5czO48{)EVT}%VKZe~r;U-hCOBuVC5W)~S-nBa(OQT<%-or_TYzMK1SG2u'
    '<iUOW6{ufElNK_XqYd_1acW7(>L+ra^Q0M19?5%HGW8)sl}y5p&KykCyH7-Q7?G^qg;|<}cNF>1;igV3?mV1AA>iZ|6AvLmyD|&='
    '+_HADu!)B-W8H~K=>to*dG9(H%#dA>b3=yZs%ppMpo?9DG(Uvr*Fl>m;(E?H2S}Q=GF#C>=X^vIY*%y+I-u4hVRTBY!ypLT;xLpq'
    'vt=pAu?_h{(<R$LdxpGv7+^SZ3U2@S{tfEbXb;6j??exLN7c`TWYTKl!*<Ir$yqGdyhB(`Cdvui6HNCdA(7kys{4FMn0E-%@^PVp'
    '@A?Ie{C>$~Ou<dYYTN=cn|qAd&<U9_5@IteM+bG-!ySVU?OmOVGUGUI3hY6n6AYlLnnN?BJmwd2nIRd0I6Gtv9Avc~;lWTRyOLze'
    'i46@IO{39_mcA2@6Pp4mVf1kGLy0CTV22x<qKH!%8YFZII(6e9LLoqdt00NHHs_yBjf7%@n1OF<u+WsT%cct&FP&&<ENK*ZC(^!&'
    'InAByQHe`8)mls{A}?RFBn@6!sYLU4RZUmJNhX76T7LUeN!4wJr;_fc;bs%tQ(#ECVJAW)FXRncA+G5m%OrD5T$rdo6uiX)JtJ@$'
    'NU4n)G-IMZ!l3a0h`mEmdunC~FCnNNC1qr;po_noEq7CL2u(C9vu07d3z+1Ad|pT*Mx;4Me#8NfU)LqkbFgkS0#dGg#SS$r0X$OI'
    '#G7aJhD>ZfyvJeq6%I1kOFW+T*Rb?`WkPfu()_OV@-XwddB34+c6`g8fQXME!j^ql_X}G4@Bxa;!d3tZjA5an#@=e5NIsO;bcxjO'
    'I*yJe8=GPIoqUejjBHG<iLyprS_4^iQ_E<Gyqb(MydO-?PGrC|u^T3}>iwNC)3y6~aUwt4NigPd#yQC`%%VyD_{oYXd=iBmaqPx$'
    'nDk=q5o8He=KP^AJ)7?{t#wQ&C>cQ$zwmV0Q!jr!J={MzVK1<N$WBCuZVQ8GMcrFIU{utN#zX%O<3NdLXf@oZ;>aL)zWl7x`n|^`'
    'Q(G-<zRy8MBbEv3c+=ka+C@U)dzcD5*S3dm8EtD{E^C2v#L4;doOkZGvd?fzmaVJ>nTbOzY1gyNrG0|vgI&S_obo_w2+w?43QHAs'
    'O{!^MBA~LN`9GgzJ#y2XceJ?kj`oV)JUXT~>Gekt$g=)QHsz<?Tri6iHu4rB=9u9lYH%-`P@C;FZcbXQNJu$K>zU=H<=gZD3FOmf'
    'FcElU9f7TGHc|F$5D5FOjsju@6E{#pCr_77s!dzvLLcZ(q7r=|dztdUI5I-cOU)EVpDmdz(tbfab21fuQiAwaO)AvZRx-^+!GgUI'
    '5RBm2%vrLN>3NA3BpWrHdr^{H%bTYrm<|b(*(7T6B{R^O_du1Z#36e{@x<vb?o6aTbm<jnf>tEr$>XS5j9NBP8rK}j<-|^QZN_)u'
    '1idR3-PxU)(-U8bCV=+S(edWxURtHi#v7#inUd2RyqpnBn24R1Q-AJdlcFL)kIf<rB@*jnB=x(_?w!f6nU%|~Y*d~;;c{Xd%n(;v'
    'b5Z5rn8@OrL}jL|+NdDydTSaWbF<N=F%hZ^gWDD!CPn>lNjTxpp{`-w5aj&SSiyPk&yL<Ydf+LlF}AQ#<Ecjk;US%{&sl`rIC{N*'
    'a;n*5T{~KqCQ8!!+gt~6Fc7=1ETt5kyh%0+g9RsCd=j(dVsqH0lgeP?1|Tn#OPUbV$yJ+7@hg^5ptPv(7^i$!pOXJPfmVxPZxtRM'
    'i5RRNsut>pbTI&%`JQk9wG2FAL2AN}fQ{beRNUoyrTMU=Fq?9%^5o&(qKeYgOk|JsD!r49woF+F>gVXY7AJ-=*;noumhAlPW|>G>'
    'jC}F8mg1Jw(JL(;1;PUyok-G@Qo8@Fn3EyX3uGRwO_LjZ?jWzAR9i@%kXW~L0*x?VE<wlsKN2d;1p-|H?obZ{iX7q0rxGa4F|dpv'
    'A6a6YBdc6df~jiF&n2f@P7`xvw4l}w*Wd=P@~~=Z<!p>LJfiU`TQes=yHC!d1DlE~EwGd$Vbh`IYF!-JQxb<v`+V+}pl9NOmuc^g'
    'JDg?6O}q2*$5;E#BO!|Q$GI`EpGNdi=;R=XQAt!+!eoJ*7+BgJrnbDhHPh_XlWA)@o(8<V?OlITd9jk-WxgLoL$(&1mbP$8e67l9'
    'R>)vR@;7_uNNe=^+1KY%(l~=SDL#QNp{TQPcRB2GxJ=21;b9MKp|OWRvBh`O0drYKXR0~E&cw+uj<;G#@8iS(mGoCO&7odlH9Pmh'
    '7i;>A5a^YNVTEp=O)wgEMLjV`%ynCyO9d_O2DylCs@YVrcF)i@4m-3ex9J8WaM2cX>V%sA+OxrB3ob<bF6uY%yJyw&QvN{Xh<&tB'
    'Qkiy(^VIC!gSc3laVsPP2ypZY6yw%+@yEL@;ENc^Favp~HFdG9BnNyq#|EyCNDNSv;A(+W%mdU*zauXE1WULip-T@E&2j`Q{H8mw'
    '3+}DN+}^}zrGP#PG>y`gzqBxJ(xddlHS&`>A;3p(7h`}IjYj#2!BSFXmJt!S8H}T~kf2*Oi9AdBCuE(^;2^(Xb|6wbqlR`@(gv)?'
    'xg1Wcz=;14P^2V}8M8&ez(=X?fl<}ABm)=!ZfIn8;8*DuOd;ga-k$VR)OAlmsMa_V=O^WHRUEC^NVuq}Eu1q4@+<F&F1}&}p<Pu}'
    '--PRbxJ6E!4o$g}lE?*4p_9TACrj#slyt7lImCNc(IhAHf;)mkdSoFq`6;FvOf#mIWH_oRmp>b2c>Xo{tPH&a-n`0EY?st|x7lSg'
    'NtLoKI?Id(Yxrhlh(#JzdUL@_R7mEArO_H#<|+#%SWYO*V1T!!fWa>ynvV2Lv}|f=rul-kG=;>|N)wlDx3XNZ!I)P4I`>+U9oOw8'
    'bvMwpwM3c)67mM@mV~hb!v)ak0_74w)>PlU&_TG#Mbw*?Q8K~ip3z~R^gGjC8u6L0YvIPyvI0RMuk$S$%a&6~YCK9M!jpF=M~R>F'
    'H)ElXC`tiQ;y=%xyJThsJ-iP^Tv$4tF)NiLNg5`)=N2@FsF~ycomlisAPmIGErq0+9@A3Dl(3*xE?c6H*pd`du0*=f()G%gu2;A8'
    'DZkIY?D-Lkv|M;ptaHi~d|;PAMDYSiyigFg_>(_~a_gL_UjMsO1jpSVyQ)zyx1*_MND@?m)>c*C34Y!<z65)g;Ln5|n;aDdjrnT~'
    '>xij}%+2E2+PaKoSeT60Lb*kxlMk=&`Y_Fkm;U#AF@#J37z-eMqX0aMKmiX)l&K|xg=pCG(D?5MMrQZP(Pr?H&YL2XOnYiU16eT6'
    '<<vpGSRCLwN$O22Ek67tC%3C$E>boq+UXx&x2!+dJ-CF=)p*DrWrppRWgTc0dvU1qhl2xPrsJfDrTEOlLD2CzGeno?D^f$IkTHK5'
    'KCga!ceKEnGQ`1SJga`=Wt};*dUBcZkTuImPN)2Lpip_-vkoez&{nQ9?P}$etoh^yyysQ#Um4YV&g>-#S+t=IY#1ugTO(c02nMxu'
    'OTJiJGrc0fMVl$JHh_z8FaaMS)x!COw~yGo%ej<^z0Vm3bbF3@z=j2Yv)taigk_CQjhc6C!bgCHO@u{S!e<rT+57OOEICrRvD7Wc'
    'se8Gnjba%?0xQO_Tf=T8JMoTjXwgF@j!h@yF(WNX3VsCRodi4V<t=gA0LL;V^JA^c>#Rg6?xfN5qBRO{3KI6PxNw`76P%SMnHPsT'
    'kqbz_9$vN9Z$DEzQzvBXhodl#I>s-|Z0o5maV(xMfqE;3GchKJ^y4v=QtVUct5F!jTSG3z-Oy%TOc`No&mU;hPuH*p8H0y^7pt7('
    'lx<H19CMyacHD7<C%)S^ro_$&gE>2bIN4OpLWv1v(He3Gznq=2HKQ7*6&_uptrS_o))eX!$qJr9aL00hTdo?0e;Uk%R*7#9fsMUa'
    'hGCG*u)bTh%HxM(AdRN}68p;~<J}q}Q~tfxGE=UXJ8VR<9Sv`_e(E`G+NPbkEC-!onL9ee6*h5}ZN8fameH-?gYS}LuyfDitSpli'
    'Wn7*qnXbn)x$Khj%?f_$Q}(D@Gc0T^c&`WTdkH9uPUCkhgf1Prsk)b(sC6B0@q&-9`Qz*Re4L6wF!5UD96NI7EEr!_61rOo%HrM^'
    'l7q7QwUC3il%jjNj{IHt<@Me2c3u)HmYP`t(n+<g08|8-JUsT6Hz%-0qpPK)6juM;pvvY{&m!#to-S}IcSS?c;Qwso#WJ*7(dqS{'
    '4zF0&Y05o09DNZSf42@*4)2NNz{w>Y!n(fb$)5U24QFt);Ic~*F}WW2$*L}f@{_+RMl}m$jwj_?P|556`Fm?MaJ@zYpO(a$fU;?@'
    'tOz^V%Q3>V&m2xTjmBmC6X8ZPR4n`GQ-&_*MJC>5N60)qLeR`wq}lAGW+sm%bJz1C?pn8fj=d&gSKiK&5&VpEY0jk`@+hCK;;;5='
    'GW``260yyZ6oE%{MYzHV>X8YN&_&EFyiKs018y6qW)Gagc129H87`qWVF-zHJdcfQDo@!om|chEp=*C8cr72kriB;d)@0JZiI8rb'
    'n^xBIyEDk`PlvhPFVO8jFVyWnE!Zs=;(AxWTP}N-6ZDp$N}N{UY8J)el6@kLEZV11H&AVzid^fIW0?Biz45<$<A2!Ra5aCmO4gMz'
    ';~XOBUal!^|3(u0E7}H^IF{rb;X@;hrZaj&9fNQvcvoWWBQEgQc-<pz5K2X<*u%OF8cA7#GEAsXtWBQXf;>P!AoGr!28d<Vtp7X;'
    '@$pDYETAqZ@f?TPiy&bw5StOyH&~GZ&EQ+zDJS@DYPna(6mb9CX;YuFmn8xe=S&g@iB+RWeNTUQ_2!u8Wf-xqel84iA*SUuf{ygl'
    '-xsydU+L(Cl|uhqg#NkplgE4@^b`MP**b6TL>dz_B}w7EH2R#na^s|S995W!H$|;K2}Wggn9p%H(=b(gMP1QM0?ADNfJfpZDOmwT'
    'bk>Oj+M(nQfCpq7@i|!>vf4zGWtQfvIdnG&7K5UV66-{;+NnhQ2~Wq9U1bBvuItrm8at*f1nk^juR%AyU4QiSiRSII_L$-Lx<V|c'
    'dPqe7;X|5d<#ulac9`e!1~le}mYp;^g?Eoh;{HiC?W8;4Q_31KkS)l&Dha!YO<7V>2uFK{6Y1%M?3XI@VK#6!oy=$xnowT>op`4b'
    'gqR^sG^Yu*H}831>~6Z5)GbTAD7?8NmH?(4^z?<%-Qx2pq1ZE9R&xh|65sG_OZF@d9W0r~0sKzNh#2Jz*^-&p=zX<sgN@?z7t-6u'
    '>!rcJD>>1UocC7=0sbtG7qZ^3o#lSrT=wav(lp5)&K9E0zYd`5gEUhBFXDg)`ha>yblxEW=C|LtFj`CkmcHj}OP2%oq7VvEAxoPk'
    '?(zF8goI0CtXxv8`U<$F%@;50<uz&<++$Vq%SwZF)i0lNrK=Db>K9bKNK0Y0CF<KZGGsDDAO|Oe#}5WeFq)!FD8BBd(e=y*nqmMp'
    'h?z|ZHcK|%8%ABkVrt4~%?=}50Q?Ziiu`3Dih6N)b;hr8H5<St8eO+=TPhtMN|j3k1*>xs9^kp#c2)pKfvxJfR)ukbswFKvUDZPE'
    'T&u*2TCg#>I;vLU(Ilp$>ZgJ-$JH8pFA>V#Am}TEWRE)+B@UmMXX!(39CR*(JPawKj`VHK$4`*It34^26v^F05X@+vNvf~I0&5}f'
    'B7UiyY8JESi>cmH=>*#1SJ5dP&K|;Z`|M$j{@t;T0CeiOWA6;cowMJ5JBP(?W3b^(D&pcpw=w(_^Y7pRbgZXV@pugQKmJf(@g88p'
    'BxTTsNIz+@xNmFKfF-%Ph^T86TGy#ZqXfhXMb2|H5am_=N&=Y40=2376Slk;5(T&D<Nfpanc|OrTBCY#iRmn7Pf)$E6@nnvc8}MS'
    ')~xTSq<oiaFsy#P0$ctn4&U+p3=)-$2|fWc5>cF$5(ZFRIX(iOYmPPP0vzpEj=9%qNs<1*6C45$C+S}h-J)LjdoCZSd$}xKt4Aq0'
    ')XjJ^X4(jzZ(sPtB_cL7BC^>I3tU-427zV>`(rRe8kgB~gc)iH9kxy+9qg-TKc5J$`31$f7jgvbNR9xvVxgNAgO$dT#vlXLPGGzp'
    'ij%Z>Nf<iJsfH0UtfZ_iZ(EXF=#xZLhn#x@3npP;BjTSu4rG@~3SgcQMPMV6^l1Ew`asLinPE!#yfA389fl*@uTB!PD+x3-Ikm|e'
    '20FE`O&HT1-<d6jh~&5%%q5P>ZcQdKHj&O(&vRzCLJy$gI9V$li@(HbC!VjC26-PHBTIxUHpz0|j0p{?$6=yN@@N9B(x_kw3}mNt'
    '$4`I*+#Sb`F-PsYgQVuuSatfD(83UMLt`~J-WZZ%qsaN<nekCZ;t{<@89K@7fo)F<Ud%p_VPts;9{MBeq#=1Y9y!(kYbqJ=4>M<j'
    'BiQ^&>J|YFy3QGcl^0-1l}rg@S}cpRtNvZv-VtvP;y`NbCp}7u`~?EmiIOFN0dYnSCLvM45}ywTsbmE*hM_zCDhiz_zKqAj0<T;g'
    'Y5YIVvBI^-S*Q3idm?&861bC=$Mp{4T2yoxFg{S%%{)F#bb=C%4ZFF1^#;=JXOnrU(Ax~2W@SX!D(_C)pN>4Sb{Ku2<&!)>)rE8n'
    '6F5_uq+2@ZT`WD&7CG87C`c|JUGr^!iEDW5sHBVB!I{M9DL=>S&lsPUz1baeDi8euadz~DxZUop;i+KpTkLKHPBNNrv(RP7HE;|T'
    '{Lb?D#fI`0F-qN4o;A9OZ-FIT_!hHTW_Fj>R`}ek(OQ~T!^HuuQ@QU-m%x&xtTQaM!9eJ(zkWd+XZvSWewY3NW1le&e?`>rZjIuq'
    '<_&Xhs>m(!<$KCJu@%(xX(P^XNQITjHNwut$4tvF9Xqn|&4d<KvgM_gVyo7KeIXrVwhoqQmmUDwFL5&1ck3%4mGTjvic-da&CJ+n'
    'uM95>dB3gW(=2ZemQ|<U{|%JIvdlPTq9lTLlsI6Wb;Gd$8Fnb0U&<F1v9+IEaiLL3x#l^uio5D~d~_h{*peb<A6d*6iCfNaZA}8~'
    '5EYLn;i5LHQUB{zS-%-g*5i_jv?8lfE6le{Bpw8dFMRNuMww?+A<?->vM?dOAprkX(xQ@rm=?n_V^rPIG?z*GnyALLn@r;6xH~WZ'
    '9KaKRsEu}69ge#wwGbILuTf?fZX^CEZ@W@|%$)dRBrzUEy{Lox?FgEp@D_Dm#2bM#qs2~te_Z~7TE4&3o|a_ri4q~|-;HqFclfi5'
    'AnxAq!4!>H^kPV#l_wEV!swh>ug78ei3-}ucrcp+2K@jBuhZAR8`TAO+yuoWN$yZb!F&{!zYj*?B?|{y8U4rYAL`$#7CufZx(A)k'
    'EQX!{g)GNOuV~pWkpL#9weERX!A(`I)E+;rJ#Id2Y;QlQ)gM2p))y*uvxc)b40>Mq@~Wf&)E+~{5frkoR4dg+qqg(7QQfA7A6M(m'
    'ZTcdPmwr&MG#k~O#*-(H8}xVQ$<wf0tL8o-Aw?$U#l*#a4IKHp<8X{8l};i#qaD;asU#!0nOqlW9~ZZHYVpz#%T~IFrsz6*t9HJZ'
    '|0o&MSrNi7wg%%q7ggq~y&`G{YnCet>+CH%CAM=NygVggq&J?~1VqfCfx67a2b>O84j1OsNgg{9$w|pct@O#TQ+5d-s-SS7Z%HJv'
    'i&?LSyn;j`2Z^gH#Ycp;IA=^?14klr)e{o1S;5MdCk!eagq#nnKn^@UYMOw7=AJ69J-A@@TQq1kv*yg^g6-m}nc9XLchv})B)Uuq'
    '#6D^qC-5X$Ug%+xa8=@_uI_Ni>D~TJbr4A!e-*4@4KFaTTs0XI=YDak?LXBjqR;QF%YG5uCujauaA$k%d$rfj6RX=Z0>>B4q~8mQ'
    'GwELh%QNZISM4QVwR2zHn)LWkVW0-st(zO4Z_YEX?`UnQeA1Rr>`nVw+xhI%7bw(F+s;&KxRpu`x4B9UTQ*zb?Km#Tm<aCNVIkN2'
    'qQ0c)!<sc8K4X%%a%iCHMT?6-d3wTet`2O<uH6~dTu}95NmMhKFW!V$*04Zul=*k~45Cx-H)aswQ$@#9GLoZd<lbJ|MgW&AgM)Oc'
    '*V0QyrOby8t~!tjkgcpkyfb~gN!|NJ-x$2if0w|)hd=E2E5jU}+SQk!@I+I^=GyP;%VMi*jHkb8Skk*C$%>$9+**0<uXf<aj&3(q'
    'v-%W-F>YEjg@<@qHf=z6<Ixk?Zza2o*5UZ-c_!mf>Tisr^tRa}>;h*&#Ebu}^S=$4?9vywSmxPd{&$=IeR966AdK@F7vDX5Dy{2I'
    'YT1@+(qFA1f25<@<MW&Y@6z!7WvA)>$w}@!Ej~?;+|#t--`^X-J@c+V@5~dYl1=BtTXH_#t47R~<xkD&v~y?KvQ<Y@_J~<>!elGF'
    'xaW(mfLE|Bd-k;W<c{}_lj81zK8wf1l5##@imtC3(%nRTEs}mOCc=<&gI~=YMgQ~k{4=Tf{b~8<Q1Wc-1k?e<SBm69yQj<|5vM6)'
    'A!#;>I6b}iaA+%;C&o&sxqUH_*qr%}1HRt<H%_HMDZO-cGr6nw($$T>w;+45L1fr^T9qP@qPdWfLz6_rb5}B3M^hem`S{(58bp_&'
    'I{EGp!gY7dsvSE|7zd-N)?(W3F0Yk{lQ-kppsOac1mUUTG>T3*&S2J-TA|z?r!wSE*g)7p@kt}F`HP$BD(Tn+ojXX2h0U}J5p{vi'
    'UR-9+)%>x#LTJ^o)S`ki9)c_<?u(^O3!x_`>W8<#oSMMu?PeO_Udjf}CnhZwY_e{`N7ypb6ILWsRx*{pksf)y2-+MQ4SWNd0Q!%V'
    'ai}e@kkLYmL4@5eCVp`!XUa%uKEj?GcW@H*3@70*5=;4rMK%I&(L34^qL`(q|7n&Y!)tQ^i8Ui*3&<gDKB2jjpmW)}sC&0*F)%{R'
    'QkJIhJ--my4oz<6yyavH5{GVxxaq(8MqS^i<HoeMM+}VClw$37X)i;;fplo9TB?_7rG{VBGuyRo-ml33MSREC+Ux+=PvE1+S`g8%'
    'ltd-6kWO@^3;#6~sBvo1G-y@Z_IWyH3392*Y?$s9C%PoW`j&%rcM<Q)_~Iz;v;O#NaK2ZZ;xk2g|IXnpaSjktQRIxDD)@QpUTL$u'
    '&C0-jy1Zqz^+yOtYWOh))x=rH<{~Nb4qaan^tOO?y&v<Mu?Y^~K+R0+Gge(S*K-an(kk7{TQtLSE~<wvuelf|2KVhcEOX8c)Zv)b'
    '?p-mIUxJyY@qXDFiR&@YOIhd@O!Svgx$LZ?amg~%ZW<D<Hlfd5PA}t`RFa&M5*U|AF()ZU;^&6k$y)l1r|~j){z-MrpJm;%>^lsd'
    '!0dGs^sUC4_1XF5Ed4T`ewmWyPpD}=Pf_!^s+wily_oNGitE-;;P+CyAM2y|-H)$-tQ@~QVteJAgM$O4*}Bmch}hbtD_ku#W9}7%'
    'pD+0j8-s0r*57h`6UIno#TF0@%=MTk9RIcBKD`iiW)aRbVYw7erbHBxvsqW~xd#dnl3M)3b%>gdTR1_@wje$VJfpXR1<kG&jbX``'
    'jLa*){*;}YWC)SWu5Kg(M-uLGDk`77qEkPoi;IQx<IbQ$P=8K7{M_YpZy27%siSo;yqKkb%Tox+geOivHUxe#F<w!=jMNc415*o('
    '{toPq_%I#*4ojKhcS^0~Vm6HH&2H*AQ$m?YHta14+pwVX*U1b4@b!1@gx;qjnf=oW|6UW%mXE-BB#~)Ok}@Y>Q(ELs*In?$dD?X<'
    '<B-PGF&V~v;~eVe*fu8xafa2+q{PpsC0><f?7zb)aP1VOF7^^`uY<emW()d{2r|BZ#*_b|SmWY>eYz_{$SRq~x&ayYHpX)+_>?kI'
    '&kt|Yif`Iq%5_At!R@G#6Iy?wtm*t6%G}kKS>A_b*`3Dsv!F~K0mCWvqJseH`P_f0lK~F+g_@;BwMm!M9d(=#;;_uc;&lCxp;JF8'
    'SJsVnSj&rxri1sGB}SS8R6#RWPkgcHv+$CJNI!qRv+Go>dT#Q$XZyZMX6<|RE*hR)JC;V;uk>1jj&ZKw(XppTAqxfH|4Rfg)Hl&<'
    'N&nrd<-l>%2e*20xtCe&Q80vBO;^)l>P%v`ogdqDnC%>&y*oW8`SGk-Y__m;ai}ZT3{kn}-6<&I5J~n{%G~;!sp0dNeXyD>8lMfP'
    '=V$akW_Lx#GUFe%;xphavCISWbM-`G=>VcS-bNoW6hLSxMhE3L?_a-p|K{|cd)#DfI<>BWvX6K-6H}`TOTcS-mIMRnAlW@S?DoZ('
    '+4^W}*BQ=~l&^H&0xxX$9tVs~D$+EtNsf(OPMV=>uM6+v9m;W7R_1ju_xW0d7m;p%DduX|c{FBHa#a07SEfU^h#EoT2en`B0oqbf'
    'sfUQPWtxBSWoxR-GPO1_@zU^YvguW&R4Q)i_G{alsI9oAnwytUSN&cs)jlZ_SR{E6-KduuxKY=^KyTFxXF#(gOv6OcET=}beEU^E'
    'm=8^&O&;K?P*TksF+$AsvKo)oyE|G}w)<IHIaf_HsIGB(v8t|b^>YpIjRY6JsJ+_fx5o)<U1N#XrONquGmg1_(-`SCp$4j#fG-Vy'
    'S0?fMWl|+c7E)kM|E24?zOH{NVQiUlZM39yG$0+)YE)5X2U@Et{9fcnP2ZUFjavG~M{~oxVf2)Aw{Bjm>4xg&wVH0Ik-qWK+%Rw4'
    'xY2`_c0@uuq7361xf#BSi(={&b2br=7nPJvsbhFk;@FD#@I{^Qf=hFKolq}~Aj&v;7rBXN-zag&Pm#-bq}`DmS8zATe9<MmaBID_'
    'c`Lm|aNMzPxPhN+?#lZ$uwJer*6s2xipDV)H>rX%7i!HSNSdxl0Qy{$T4x90(A|X<kdtIwT<QugXAN_(>mOT4lIh4%87n=ARP#J%'
    '`c#)U_WsIac%;wa(bSx(qiLbE1Xkp^&;n+4wwTIs9CyE1!whza2sRNB?=8wlgmX9I$rD2u$H({C>V+2^=_ExOn#*XaH{>Pzyy-hN'
    'eP?d()b*W@_D)0JxmnYYwFDqpk~bSof-Yi1VMn5!P8{jTo?PW`gdy%@$r8BFcDw1#tsTBS&MRC)S}c&sDl|=*e?Eb2UjFoMx%EP3'
    'bOEP!Oz0vbFOp=@>LH^6f))bS0$xZxv-GOwT(O2Q^IY!WozC^N?wO_FPkp(0k-O_kQg#62zhD6TsBjSSq(GaOq}WEQLVpq3#1_RQ'
    '#X5RY(@z@qN!{{;Ns4=hDb2<S8Pc(itI7Z0-j{c`Z6y2t?@xhf&NzTX(F7&S8p=|Rw<JE1ZH?vR&OIK7hYLtC;%0)Rq$Hlt{_0m%'
    '-RK5DigKLX_ujd)$OO=PS6A0wr674XF=Bna&7lY&7-5y3@QeaTN^_o=+!8YF#3z*tJhNwIG`h|e{*2SfZS;r$?uDc4ivBX&Sh3q?'
    'H=E1j=r%z#kFLw?kVYEpN}DBWA5CRtF?A{Pl{5ZwbFH8?oU*NSa$VP^=uY6;8T{${V|ky`;|(`fA^lmvg6z#t_JUWf>!w6w20yB9'
    'GzC#QeV8Z4Zsyyw!(iST46CcmP^U{>8Vf#VaR}_{4-f1Ca?ht&7A*Z#U#u(hUVbW1>h|XKivm&?)xuCD3yomH5oy^!U*z|)iDIh5'
    '<ur%rjihdNv<E}8mBX~4q9`jRKen}p9jSKfjukm*n$ZMAv7oQ*V`=cnzKq?MMJroNJhW?UIogh0?J;LXLr|MmKW+97!=EY{uBb5a'
    '(&tq7R1b+S<CX5*tj;oaD5+5OIHWZ};pzL>ykVrix}Cb$PTkb1n|Aihn?2LcEoY}>Mfq|%oGwFM!wT~*;9F>s=CVjxvYJn~7?D^}'
    '3oIMDoC)2=D$2~+455yK4warulp`tP3Z!j*EPloue(5KPLDaNiYccLzG`q<ZndFH#HXe@-&>DH8hTSt3at*cV!aQ)&#YV$ggV}f;'
    '`1XUw&m9Mvc0<zahzLxjYbLFV$k>phPzWdVsba5d`pNMEZ7M@>s!w;V&DhgmGEHW5bm`(p8bgc?wqP_eN&6#tFicEhLS8&B6fy3!'
    'KUIVCNpDWb-#8c!TfYwItS_cr4mx5y>jfDe*({7E#|9^B>`aD>(b$R=gfF39jzwN@l#@3c0(SR}^Q2nPS$Rdpm5tERGlGy@EDp~!'
    'uS{NXH<?6cxv7?P71f6QvdxA;1>0_%ZjRogL<B0tCR5vC#{A?$L0kJO`oW7M{rs}czDO6B#on@6)$&r95E;`Y%Z@rTsa$5q$)#>&'
    'G%W7mtn2h#&P#rbOL~Y`_7Io$5ZfNsJBO<=(q(F{;>&asD)FMHWPZ=m`MoGpiPMU+n;@~2EGvr-wy7jUXCUmV)nI3@G^g&Q$1t-e'
    'UW{B3)H7FAITLQ_(qI>L3Fo^`-UR*RM?eNTidMA35rA)PB`#?NXX-Llioa)#5#`nhhm$Iii}d$qbsb|6;nuqtpdS6bS!L{;nYO73'
    'tT8+AY+!HKIF`Y}2Bg6XSSTm{nNeS}zZ+=d)R?-|#<jN6$hCGSE7y|k{DUKlSB@<HgConE<zxn7D^-G@Vdt2Ig|Sz=8XnDr75yy}'
    '64=C1&ZU(iaJH4dSB%sGByWG`u;ACsHZd?A4{YR9Hb94LBWuSfLO7(+uQ$86K@s){1cR-T8c+Bj)U4Y(P?$d8PD1=S32HPVbJ<!d'
    'NlIrkonI8Enti4Fq_b9VqCuo&UHu?NItP&U0piqXj;VIi!<5agG*q2=L7SIC3{D@2U>Z%9CMQ7_Ob+MX;e5A)$*;&^HW3P#{%6vW'
    'GipF!FVEG1siZNKpwAvXj2|{s4Jxpr!a)ton<leCYdH9o!$NOrw!LTAzFE?5GMt8!U)31$uBP8v6TAh+qB56xPDF~DW{xQ8&PSVu'
    'Zco<&!RW>`&vY;uvpQ5f)_rFI1vyZdXFp|ej}QY)gY8LpnclE9cOw1TwAk%Mx1Y?jRZH3Gj2AX-WI4oOotDfZ1K>norqf9Z{{_y2'
    '>}rTXokpTG4z50Oh^16yyYa!*$7&UScjAW?G1ENeAjMZ!#5?ScTJzc9IwSr3g<D=>p^VzN5-jn^Ey?4RdBn0BYdeuOtD@ae6kGCh'
    'nI^2L390_cF)-dYLGAj{!@YQc$UWBw`}NwQ!o}i3G5*NKxI#|HxSZu1<`9Ce^#HVWD8{ih@DLL*iWgzKMaL?g&Wwi()v#f$yIsO1'
    '2=In*>Be*b7DRJqKGccG*>~C$Bwz;)95bD|-R;vd?gaJi<rQ0wAYzCz5&~Y@Cl>u@o&zogC%FI`V8aHEvmhF<)JCx5(SXroE<^N~'
    '?N*9Jbqj*C(J1K-2*stMpCs7U?ko&gqbNBoG-8Z|(B*bWCB%kdM{F&`gh5QEB>m+6_60O3^#zKcIA3dg$=M2JpfHDwhY9&QWPrRh'
    'VdyVGb}xcXV50bdf6Zz~E8C1eugy@QA+Nk%(xNoyhT+hKLHW?Y|6WvFqSE0FyJpVXsH>~M7<eUX{OmMdvY@VYmEIzNIRy-a!QtqK'
    'vcJ{qeaCVJ2-VuXBn-pqBqg<@YVa)qo&^rUH;amPrEf<xiTZXQcEzo>Xw8w*L-;8PNJQ%ec#<?-V0ZOU#0O!0vD2?(!Ro(Yved%Q'
    '@B`tK8wnzXiq^zGXYeG7!K!XK0!ez+n(`TOPOpgo<eUVVl~wR78cY|{=@5YzdB8w*Im4E8{!51O9>HgD<e<F0()~>t{s1bqI)fA{'
    '36-~_Ac<xT!0~5>oS#wj7qJYAT$sgM2WXGh1bc1R{m!nFWlEc{TdzL+JCJJpV41>=EtY3GD7*3BkMiiN5$x3H_MkFRNxp;gcjvE~'
    'Z(bwi0Ya{t0i9*{hz%4r-c<ujh(~bI;c}hM#UUi*tGgS~Z9ynZL#$oTmDGTK+2Xs-WHcQhXsP5JF!)&kVlZ<ox{zRXd?RsEZIg^5'
    'HKx8Lty*(WE;4P2SOjF;Z@7Z8|9`3k`8QdrDI?Z#ZV3+RI+^Zvn1t?T$)gI!W(c?oL5`KLWkzQjXB<$kT_LvUuCb_^UI4vwP)?e5'
    'MOUqxq@CGdO7PAofUkfZZAeLAK3efUCp^?K6Xlo;yGlO{#2i**@^Y9{%;Tn!lcqi97{Nb0)xe54+U-TJM?*6P_1R!(SNi=SEDlXR'
    '39(@6!SbIC?VZ&$Kpl=J<6R@ZiGq)ZmdwhenH=ewfXko;H-|JbZT%#)a1Me`&;R@Nv)7WdgA;pLI4&#5GMOWjPX$)jpdZ_Q-rx+3'
    'F8I3>MwE(rIg1tsbe#1;J`FK#v1W_{7>sRd$lh$iC((v2hDf(-%^7*UQ$z=*k97VmS#Hjc_6KO-jAA$C#*hpd7j#~AC?E%HoM&Pn'
    'chdRx!?;ef*{Rb;F8MqXWVrb}%tT6X<HzBow$^kpTe8pz*h(CA<g(O}a?rmM6&Tm(KMU&xmJbN~c^zKb?hf`h>WppoKVmm)Ho_+7'
    'y9`4tmuZ2nJfQHsnxJwRTl~(8Qht%lxV2*BS}QGF&fCq7k@4CUI;b7y+MCX2Oc=GB&_~)gVWA!_q2lMT&SfG~$PR+1y~MP682lr>'
    'I@tHtGS@@t(n%W_oN4Xnb*<$Gl&97Ps#1k{;)$Bqe8T~&VRVg0hyd7PCgTo-cb-e6MaVW<V<u^W(tx3QpsZnzU^7BkY*`tJcL5IB'
    '`$2O!A*!g!xw{6<blMt!s5p3!#wF*vt5>7TI$3Do`m7z4Ce*_Rtga;oTse2tuz{W7%*3PH-3@en?6jWcAe1YI>9kDhV5ck#lZAgb'
    '84JaH9kBxJTxUGjnap)ooa<3Wm+?AF)^Op)B*y4yO087{o7XB1>Jkkw1~1hNzBkv21Z<rx#-cJ+53ZT;Xf%6oHD|h-*N-w>Fth8r'
    'D2Xkf4?M0Wt@SM8vo%|F<CkV-p_LES#&*}5d>N}P1CNDjkldh~-5|`$YW+gtHcI~r3L0KNzh*GLYut5p;wuqin`^glS}ba83A>#Y'
    'vL3f~$?nu%NG}O!V3O`2Q)Ad1G8I||%iv~ITZhZtia}B)TLJ`;w~1D0(+Hl3s%#CFP6|D9baQ?m!gVyVROWu><iO6VxYKBY=daGs'
    'g6@RxSL`2ACyT^@7`zKvA{oGxT@G}n=cj?BCceJm)0oKm&%wjm?qR(abZ$CA2+e~3#Bt4w&v_(Hw9Zroa5w~`9HB`+p?+p&G|U+>'
    'wHFydm|(=tVkY>b14YNNMSI`Y8+pTg1gsdg1|yAMFQ@T>&e(|PYEXv|U2rt-qO5JqfvMFnwdmgQ<LkGFM#EJn05Ea-LzVlrX8naQ'
    '5IDG4?@_BWo9t2poH@bL+(V={VB&B02UWcC=raEjNaeydu{uOHro#bZbu^H@hULMTx>zKQl^_<B#K~v+vq2XGe3}%*nn8#uM4m-1'
    '`^(dM(!Y<f0FX^0J5d2&K)dwIFFNIOEIg04P8G4l_;&|Of<4h4aE0${jeFH6fcnrJ6XA}<@I<*0W4s(p7O9wZ(hi^fRCLJ@OMwuS'
    'JCw)FU=LCWic<~VWpX3OtW@DLRMC?;p^;jyBngs!Ag(q583Ea4)`X$Lk4DogR`jk*gAUwU@-QvVC_c$ToI93%tMZ06J9PhmAb0g{'
    '+KwL!9l1CeJDw<76WiJW@Yg_Wgy`91`)!T~k56GCtr4vex}buDrs%Q~WbP51K@{dH-O?EQUX(u_*6+UqP}m2)+!k9-jFCcv?P4R3'
    'MSGa+3Ihg_C9No<3HG!4I(3+8>m1oZh!K&M5I8W^!{QN!3I!CWOKIC8OWwwDl5|scL9sw&_*11NIIo0}VMlkdCN}}w2P3f5bS1RK'
    'Zi-ml*@WH*AC%cc<*m+x@{>`17zFeI_6+VhC964x`4cU<C1}r&+=FTiruXVbH^+;wnu_SiKhy=D1$M+>Tz1?`lf4f^*Zoiu9hDd@'
    'F-d*d&2u6wkFu;tT*8W64p+BUzk72bO$9RHm1F!^Z1q-lDf}o)<2GxLlPvS-V1InDe@W{GHF`5m!?dE@S%qfeg$!9+15P9OlbM*g'
    'yP5MZD9cpiCN<*>FFj=y0^i5s{}TU*;xr%@$z3v%&a!ZRoCH#qi6ATjU1Vg8&?MQ-S>UTsIJJl|Td<hI#XFpAv`n)W*Y<a<cFjtD'
    'h%S(v6(Ecfxm+f}1J=??dXHp2A0}NN1{eWpVE(&DPw{hvaZ1sh*%smS_o5y9@Sc@st{u2h&UR}#?d)CpFH@#=YiqQrpQTT=(BT@F'
    '0rmaF7D<MdBruT3;kaXfi!TUI6Pyfh8hStJ9pb0sVjox9P26Mx2YUSk;l?d8?ID?-qTEoIbJ(4*tmLRbPK6-eB;JgpBs|!!Rv*Qd'
    'Kr#aa2a-uS5#1k_b8LV|Y|q|d#z>s~v?63`iBq&1VGQB2ENqV^FffLGVP>G=8D^Lqy(aP;OJ`FVGEp#{Bq}qsx|&<%cU*t8QCN)3'
    '7L6~ZD^~9vu-%z2iGe;%Y+g|4esI86Ylj6h7AHgm)=mBkUNCzo7(6*GKm(Pu#+(+Tk%J)toZS=NC=f}N;88uerY%*cC*m8<GNi$w'
    '*%};iVV^SuUT~sCCr^U}P-LCxAwJeWatY9>)D)9g@xd+&ParXg#Bji9&Z>;ih!LTK!YRcdT4Y5V0qkfoPxAikB@_W9fq-BZ2o~cI'
    '+<wp(a?lsXMr%<XKW0{g1ttBTuDscv0~y0^I%;0i1s(EGK$n1{x2nva2@Ng4&zm=}@$nk7#GJCmqM<zwZ1u=n?pf`@EN+M$p%9fs'
    '3WV%lM0h+#^YsU_y<CZ%*b8FeTw|)R%2}qRbDmZvUnCrX?J+7m+Uwdr^mPk(6OZPhnG;(uL?tg{`;upEOO_#nLb=#;R?0}^TaEDg'
    '%Folky1~$^{<bC))LUFRRQwxSN7^<_;caYR?cr6tE|U~T8sZ7O<*M0|aO|PNmpIzof^Nx*Fk>08W7VjjWME#<hj#RshLBlx_}XMv'
    '_%CO*&+0K=(__50N3|R%4P*4FG?LR-qFjRdN%T?cB6$FvZ4$)8=vh62Rl|jFcf?+r-i%+$gWz>Tx(0Gm_U#u*o@V92XFvJO+f|>('
    'chw(c0Tx|E%&9Y2-49;#w(AfxL{jsttV5O6R#Rmg(nP64%KRnd-7WqZKhOn<`3Z5#)U&ZZv({Ov{GC~yW_PWFMq2O=@!3^LF~|A<'
    'wqy~-XPE@#@v#wsd;x!D+atkUcn9ZY*Zu8IPK>vjBs<=Szu5R)jx*v#uQYbjk|pOh!o@dk91LnG7yKe<cryB2Q*3AM=?$j8Y!tN6'
    'bGkdc)#>gIZg(rZG;j6H%zQH+^AaH$AsjU^N%(2*yA80x@=CAGm9dUsg1{tj%Zk#iUTSU<3afDgD~D!_u?^w}dwSvp&s^gR`m|RE'
    '9%rjz_H}8{NwmE&UC&gSB*Lm)KRP64gl1KLcu)xj)uf8Cuu-y2iRc;_v(Yo`D;|mTvvys31TI*dxowg-bdiO*IAhHXM=2DodOMk4'
    'C7hEGA$~=97DA;J(J7g$@vqZ0aLFvegj12`u>{wqE2gUfMk!G~<q;p*vuEG`kl8u-NQ1*2aRJ;_yzXAYfe{}KyQP(2!~mY(V?N5A'
    'i_6>N<d%YY2CveeI?O`jXdH0`*w_`%!ky=PEd$e<7lV9FfT>@5FPERfg-MoI5aFJ(nz(b3bS^V_D2N1`L=M1pW71ZYK@9P3;5&(-'
    'fP@5gE6NcV0W5SLgifH(q7fgd|7Zpe+7v*-sz#jy!Xt4*N>8qEgy5Q7vNzc(Zy&=_Uhj9csQ0Y!tr7i^+U>^z@L3#tB)HCE@kH)U'
    'XzL};+Fk5w_hR5c4$sC#S^@=`clB^WM?^L_dwBusBrhlIa?Ow%@O-MjfxSLL91SNHWV+m9$PJelw!;kI5W$|d)WZ=T*rTWG6tLHo'
    'dIES^uX9py874P*q{HAQy>KdDnRD4}5{#AExQtk-AtNJDqcIbX#&k|}g#0p!ybFQ-E@nx?qPhMoxe~~&|9nXB1X^gBm>~QLegql}'
    'fzGQ1qD&@EDggwKb8!CVyXe@L?;6u3iP+i@!#Fk$P-6<$OS(Le@Hme|c!iKwJC7rHm7-P2Ft`~wm3P$Lo*JfBTHK1hkNG`j;jM7E'
    'N|-7lAy`8U6ShB}dpa3MmlJ&{|LsZ9d)s+cmsbSH_SIhY%HecaZ0NMf&vEYAX^o$qrhMI&CkA7aQO+VhpG;iQmY%t>d*;T@nKLJe'
    '3&=V0Kd*DG8{Gtwo?Zpij%7ib6HKuQEN4cjWFqdoGp&k!x-5)}f@_geNhNhMW{H(}^b%QeurE>$+!CrqnDjg_a#jhP3M>{}S*bwF'
    'k#7r|(+H|2a9}yUvq~@tXv!EJSddw=NE2T`<<Jg4)RjOuD%Vs~0S~*h7KArIr=T9v%v#2X)x<=$EX7>>EK}6$?E?o<<f2vwa=Lb*'
    '&)<v&zBGYVFRhp3<q9%bh4U#QMp$RX%gY?;OZ?yErG{`Z0RnL3Ip=7)l&Ap6K(R{@7$GZ$^D$ocaev*%hyKT?d{ptOQPs2RgpzV?'
    '@2&fIu<qj{|Ko~L)mM$`aNUgmv34QrYrpPov|4+s#ueArO*)R(e%)XD_2Ie|bHzuV@dBNiFdvWJr`!zBG+xihG<+Tg)SS|V8?!P6'
    '86lr{D<?dA%2sx@NAlmr+U)d8)&);Z_`Myf%?jVCiNBY7Q!l*Pvu|$IzMw{2=qE1p6Bqi4vwmRAUAsQ4$$uAxh()VNGMegFurrJb'
    '$j6dh#&K5iLECt2UyH8_(I$gYtD=#3mAvY*`Ke>qBZ_7TfPI6|5iYq==-0I08;l3(g%BM8iG-tDxOiDuO!RjKflVyTInia^R8qww'
    'uWsabAjHx|o>9CgFp8_u_~!hf`Ok$z;$7T{PqZyKwBW@Z`NfihoDIs-w)6-cS?VOhaJ$aRL7Y-nZWT>?v5Ka>D5YutFkQRR#{HF~'
    '=ocVIbIWr(p}7IES&qb4<+rd#0+$;a3EaU<;Bu{*z-Jf=d``}l&oG=QL<5F8!%~pyt*%yYMVkVnL<u1#Sq{=cduZ8MCYu9Rzj*ko'
    'OfBPOyKr<NrQ29$|CLi}JElmF>c$iayHsJQtOR_=Rg8y3egL=DjkCHyTVDPAKy#)+D}!0CqOC#jLq3Ag?NJGxeH)Gaw{FryJc1Yh'
    '`e%4u2|h-QaIT#_VZLJ~lF5!l9;dy><ugm>wEvJwPGD@B*-p43G651?b##B8ft=xsEPUO19Xts7tyvqo7hyTp(h{)WkU(uPHo6+5'
    'GJrw9uf2*`5Ioh=SZtg>IX@@fs@0JTmwf|wi7zBRvz=KwUM$A_qWN*0J5@(VnHrDo<%FYHuob$In6nFnGxsBBs6_u!hr_~+UB=g7'
    'FUt9v3Wv~ezsI@naqjzm?t4A={bBC=UgX>F7Wo^uw$I-bN4J;zo*&&o?t6Z8k8<BjM^|4py2D%6?jN_TU5?u+4ll>>tQg<ks`2Ic'
    'os!LsZ&|+_*HbdWhig|bXGsaULKL)TNy}xw^Fa<(8aiW6C|@`zp|eBQ6kSXvgd|(6Pa&KE&M0;@MV3`cu9I13kopQ2#wC+6qZ+Ya'
    'k4h-2JMctnxFS7}E-Rv^Apfdfe6<P3qAR!o(^6dCph$cdB*@m94oWE?oTOXMbCz*GsFdS=N;=H*L)=x(6RKMFM)V04XH15#$v5os'
    'E<!@Pe+@bW){88Ipg0bwEV6Iyw?p~%53hn8Y(BqH2`i$ba0vNaqgs8>YxF=~J}MhUY{u;mJl2tOZuSzEXhZ&Ga}MT<@-SaWI-ebg'
    '5R}a^2j4IkZ7Tz(QR^m<)X_v|Ek=tW0FdJ`O`;;<P_${um$3By)yDMxo$3CoTgdg#rvLrwpKj~ne~c^q!hGSzoZ&{iVPMEwS2b8F'
    'ODN|UO#))zGGv5d`cw7(e_M=+vc_HPH4Cl@cC3sA`O@nnt+)|9CF67g^LmXclTId3FmM9#G$^7>3&I5m_6}B_fqVU@a&Ven^s<;-'
    'bTbZbHJ7(;TZ+AIBqh1+0B>-9|G?S(^triASF`dUL6W>=Y!y$ulACpyS!&SZW|?vH8!0we;A#t+zo1|$a2qh^&k?(kIaZTd@EHha'
    '<<7{m4#KxJO>K}r&es$b2Y#Co<d+7YFDCf;6GT5b9S5-oKb@bQzI+#X2CsfFYBzeNIg~p)636A(24?xqM0P=fZp3i;`52*EGT1QT'
    'HcD-YCI4pWCyu7UX2#79B5ZIVj%-dg<b{cTaeb5vwz+v6ZT!EC$k^O?_A@ufNnN%2$o!EAk=-vFA@Y+Mw9SSf7CGN!$_SdW*xs08'
    '@tR_4m19Hg*Q)i0J;F*#Xsk?`4eD6)jj_qrf+$JV<%_;1nGS}N{z6Lb!Vtpr6BqhXcT*>8l!>K2N@jgx#jGKY2^V^eY^Vu5h9NSl'
    '|15b398y-pt?$)|JD!XNoz&)iWC>h%(wwm;buOZ*cwz-HE^uMWU<Ssc=n8qPC2A(ktC!NvdY2<J*B&ewiCXB|EQOI9_)D6fPqcik'
    '5>t7O4+qDw2!?c%MgPK&U6mPKP8RcY&`pBQaKR2i2sq(syINh-6U_1~Z)v;bIJ%SBC3{>M(SfnCoEC`va+hvBGuLSudXbMOCNNRx'
    'B$d>x-he2M0)FTi%t@k1cTofCz@+||xwqJLs!7Mp>$~SK_zjKJjiE3a%Lw=*eA_GgM#m;+byL1R?tE%1a-WPiU3tpf_)vd4#&TZK'
    'Q8D34A&-O<<w%VtmrA(-pr34Iw4xo3Wjc!2wGrp)XPH_?S$2#<G7t8W86w7XFaEr(%kET%={oi}FzLnLV(l<rH}*H!sk^tL?xz*@'
    'j|RDTMv@kL114TIpPgobHtu-ZSQ5APwB^d5!paGeI!`4QKjMRBc?RJ2$CRF+wloNz_4<v*c|B@aY-d6cs@yBBNbDnnE){ce%nl%)'
    ')R+UW(gWCSon9Eig*G<E#!fHXoPZ$;FfJ%x635(oG~iIdiqY5Jw=43P7L&Sh=GGa9!K{df1`<TaN5zuA_}PD1g#>3Tb<P;gBi}{C'
    'NeCg?A_{2|`laV_NLQ(mPSMX$(yaQa3KanF-{&=1rNIh%21Cr&VTc~J(bHm;Ag&+4=61DC(_n5JPhG4AB~R&C<7ZXWOfcvKk9SY-'
    'F-ui?xDC!|z7cV*RBy+a#dV@h_4)gXPWA<#(7!u~-<oHgx66+il7QYnd-@(*ry=tBSm_Mt7tdk;LrEZf>qjgtg_@yv-wBQ1VLVeY'
    '$$R%DeOvzXZMH&A?fUx0_X?lOJUUC2h~w6LF>4KP-`~S+wmQ*G?rgNZ>v6utaP{qg19Ft_RlT@Z*dWI@Xq`PtU6OAR7WyXG?$$Yt'
    'u-UtMXhP-Z8`$RYxFx5{>NWLf(=nTT8@v_t@uv7u5^Ll(W0mv_#~w>7bjH~!ri=S5<dK`_1`n1KXb~2nrdMR=SWD9=#n}{%#N*SG'
    'a9RndOuNFK;+0_BgpF`TVr<)(YYCepTYB4Gg@UXyAvFsk*gU?%fwWA3CYpeTuAijyT^2hqT`o>jCZEzS=L66bLml>SX>~4==yTFq'
    '=bZ@GUk0Ch6*yt}E;t)`4*hSA{J$)2SK<M&N7#K333%CuJ{(+g$eZ?*MmnvI@e)~~K**oc?_aWvKNpcpYpl$LeGBP#7c74#3{P;^'
    'T<(W5uTlh_p~)|Sp1%Y}&J&xQ<y%6a8%|f$hd3&yhW(X%UmTkc9Q>hFKm-t^pKh`3<A4L{4kSrk@H6&?qm(N@nYt0wSnSUAL<S%M'
    '2$Rl&oonPW;GC~$>N0^F!JJ=6vRC&lo<sH&I0R1KXEbKwj@PQyy~rnnnASt3V{S$c;GLL<IUu2)cN+q!)Sv~Z1a)e+TE*X;SR9?h'
    '`3_=a5{-+HbgkKZoXr0HHKLY->}9V`I&*al{pUEUR#&*()R@62y8&8p@sq{>D?E7GPIE2@YLo4_{Bv2eBH|g#j;5TLlEa_v#ER0f'
    '-uaLGseEWD4~^=4gGo0y;b9tQ?~vW19j}<z%n5ybJ(-oL^wXLcZ*bbQ^(A9B3RZ5Wbe$;V)fwz%$Cb4a7hJQ`wC!dI8oVr_|6MBy'
    'xxKqf%QQ?n^R(i1jGIi3Bge6oOD-$nVIxk`fPD*P$v7WKWD-AQXQZM_I>Ry}^P)bYd@DAO?V3i*(8rh~Wvt50<<k-vR&S3e7+e8E'
    'z5q7C{1r=MID$(XEOkia8RwrNB0^HiMC?E8q&{HbXL=;7*7-<8!q6Zy4jI9PYL1;8nsZV=amuJ+!jZ!@Hob98#heJ&sjzUc9L#U5'
    'xQ$xi6o?h53ZMP>?)1s|IZFeRQEPfJVV5FqHBylyC#6rFh`(&6hDfi*^$L^4=%i;vfgSsd>!h0ilD`T$ZW)IckR*~3#~OImeD>ze'
    '8UOq${K^#3w{8{<#zv-BL<D7=zf3eRD8sB-A)qXd#K1lg<NOjED?DdWBSW!$70QTz|2_DMf6*}e(}kxJD<d7*Dq-ymGa)JmOmn9o'
    'Nf2ctO`QEfYT>7AMm!v1%&?3>B@x`>r3zh>EMJh`q&%bCqB;X%F*F+BpslI31`2f$6{M#JCXCeI3{u(gydnN+Sr#0wj#cL|DZaop'
    'YJ2jIgY~CQhl7rlx=Yw=FCc)xeRsmy8)%qP(gYJT&YIFnYc`l)jFLI+FRKF(*N&<t5`kZoGA8_h4l|udLWBYI)u0{-_SI3=fh2Zj'
    'iwZ#|F=a&b11(A~A`Ue5vU^=Mnj-dzKz!`uphsOa2chr|z$=`fhht1AA7MoFvJWiaN7<~Uw^=i%1`%=V+zNi#*qTnvG>y4y*9&xs'
    '2-34jixZMFY1!RSa+_s#sp(+MAq<j{2LF>(4+A25uHXUcy5QeP5J6VUz(=tWaEvLotgY8-K_?lf3zKMqo6Jz%l!zRfi_vqj(o5*6'
    '>1@)UwMG&jOuAd45##l!$nUq2FF-<=wVcFS1%*&$nA9BvZ}iWY8iw&N4NbIrFyWX1r5<ZYx8a0ml}sn#=@@ToxN90MD~U8xHX}+o'
    'C{C{vgQ)0i!MF}3-@N<&#alCwAY6>Gg86hTG91Cv?Zs%Cs$z<Vks>U1d(yqp&<8E2OF9WHST!eE4|F#vW}L}}%m!0&Z%zU}-+Xl3'
    'By&+w(w9s*=>gGDj)4q)*bXd@)nRX(FiccYLN|ZCeux~Yhw(o+YEye!B12T!r-Wtk1GwNb(dhQs2M0{5s<^*ONW9t~PTH+uwbkub'
    'k#FbkL^QnrJ1l4aUO)QxzsC$d#{@FeB!b0k4Y$G$z=&Q6^+WGOSG83PbPCu4PrjoAK2}+QM~j<vc<-(PW{<erSPWT(4>pxRu3CNn'
    ';iCgvB$;w->eX4sIk1h>&iUEP*Ufis%bCHt%hUPmIP6xH6mVD3T|sa4mf6~sE|^)YB|@v1H7c5~@>rjJ|KiyX-s@&_+!{~PAz_5@'
    'V87J0!&hCbIP(JWPrBXEfz9@`;=)<v0a~_@`DL`c7|<ba9awoh_#W0@WL3T4u3F(+*SB<avtO%Lc_SHGCjFPp2H9P?9lS+S2ODZh'
    't}>gQ;%%`BAS)Xi(?rJ1gj<|S>6q&u>C<EGz&sH*{Q6z+08X>NWh!hNtAbG-BSM?<BbDo|1jWrDOVefrrZ^4UQ3($(_x@~VMpLd+'
    'L8)(03elCp1XV;%X5S;>ET=;$Mxop%!nbv03$II%42k}<`jKck8ny_y+Adc!tq!|Q{XDIHZ$fX;&&y?G)8w)Y3B;fy{Gy+{L5L@M'
    'i^#r_1sL5AX`=eST<G24=KbM{l&xidKILkw43WOr5A>!fAsSSgi!=U9g}JyX2MVJFtOPg0)0OLq2V=;|x$zU0Fn&V^1k@Oj!@Y3H'
    ')(2W&#L&56rKug%-NW4dMr*?1xmxr2xJgu0v)zjiLVmf8p+yMu13=eT_{Xu3l5guJ38^PZ*sw<Z&NcSSmv;u7IZHpz=Dm<wLdHj0'
    ';|GZN%L4s4*?ri{S0xO!q?JGe@j-oC0&cE21i7*&;eDK)tVKCblco2915pO7zg!on^8RG`d10N)6_@p(KWD*@9d2E<Ed6@UDzqS!'
    'N(cQ>*q>LgBb7**7hZ;Qe=!hj6y5Q2TDR|;voV1f&;Y{}ZP7N1ru3Z=w8?u<3N^VBa9tbT^Zu+}xK!lU0_DD<uHZ4GH7mzsg1jU_'
    '*aip-BD-ZiI-^fTmC_CyP^Laz`sJxop9WY@p~~O$YH!%;gIO)5Q8O{tD&dASpo`!evKbjr1IX5itPe@xY{Wc2UGbw7PXHH!%E_P`'
    '_FMA=kcu>*N5TN*47ZF~8{0}iB<ER9^#<eSxD!jx4?*olO9vcp(6Y&8#M|naL~cxD%(6Vydq;Ao3*EvGE`d>g%j^BFce~!|fA+Qa'
    '03Ne*i9wXvHro`98nlQNlBac-?^AW8>u*QAakPd2`JX2}?r_;dNBjTzZBJ;eGiMcQn*Q&F`@a|NnhW=T=AQjjyjrb*!<Bsq+V8ge'
    'kY2lG>DBu%6S%^lU^eJJAB?JCF|#i>&X*hdVn6&ES#8b%`ipaF96_LN+FC!$?jY6RYbL@s%jBGta;x$l<$YuC)+&$9H#T9La-@ud'
    'PC-;5R~M5s5k&!{un?SQl6wwAAX0nC)-N&(aFm1a*{BhM5Y6pgu^~1%WQ5E&>hW9T`O+ur5U_Y6+TVycz@?=$fpqt4GRxU>N7dl1'
    '6HAzGUz;FM*Zas*O>rVzN4^SCuMzWB|4sjlrfgRK7<l8fA`CnTAUqRg@$Yf7*-e%-wPq6scQWhHsevsHJuiupVTDf9EJrA4;;c1m'
    'jnezUw33kn+X0J5n?@$+#PA-l&Yy>$DG}gR0%5!jI4?RYO$43o6m}6>4wRL+Qlc3#6o3E%U9-7V>fa%~fOB%(vtoY#n#<z~Y3?68'
    'dM0V}HqsvYvf(`>^L1PnLZb?0m7dY!8Hu?nKvh3#bf3{WDz-q(tLPs?y_4);W$l@+R+8MCL`W(>Z1_6T@j9_TYg(*OzC^Bn1SN7&'
    'e6ND~f{cE({k1Q%Uq!xI1q`<8Bs7qlaX7G_Dr{e!v(cz*`-^fz@hWDh)mhE<WI}m1^8e1E0_>GzM{Py&un+P@@Lu71iR``HfN2A<'
    'aZdLBWn{#y&aHys?>3YPZz2iSX|Fh9nWUW-u;PP-SsZ_9iVO3F6cUI?SGF#zSEXc4%2(vmS`OM}Heri)EG!P&x#S8kzQ*^bN>-&z'
    'dojc5mvnT(Txo|8I;(@f%Xg}R_)b9pFa8d`_}wLQh1+3*b=d6_HMVT>o8qhrEc9O}q<1E|PC{Rl&->zeCu#FElfu)uR8J9!h@J%k'
    '=HuVJkEQ0a>-c}COEzd0;85dQ8b!lwFc?|MRVDx87M{{>w+)bZ4gSBUAh<<6@JFO1D`X^}rx^GWf`X021b?28;I0aRyQ%~3stovS'
    '$;6kIOXQ>yUq&XekwoHi<Po1GjrenA5qFhDI5wI86Jme$H)j^QoC}2MobZ9K-aLPS4P<Um2oA&jHI3(g8&cG?*@TZFq|&H;(rCPX'
    '_peIezk2oZ_0vdBuMq3a-t=&~3=Ncm9cl7IYdoiZgqaMcF5hVLx6{>m9OSz!$@8u$XY*y{bxHY`&;QjtKYjBKRkD<|Trjx?gT!YX'
    '0u1JhiTUl*+Hely$YlBqIFBElvG*U0F(<yOO+RWt+ClfVa7uCg@ab-PHTd<{VPcYxBR{#Ow<3;TKkiOx@h}8lb7IRN%$Dq3=;oYy'
    'wNtzwoR`9+R|p)A4wYE}jB-KuYnF)6d`7r7`ntD>oL?nGp35iyY8lY4yzIS@(u4C1$4xQ)s{~J@hP450SHyNAVAD*o4CHK9ESsX8'
    'nwOd-dK`L%DT|CR&wOJ;hUwpfLt-=s!!A2DO;S#6B;nl<Ko96kB6!M>Sc5R-K%9eLX~Sr==ii;S#(l8g-#*!+&Fowz%)FvaCs9s6'
    'Y0g`1CMKz|=6pi`qmu9OFr~xRq^S)$5DYa&t?On=uZIbpj}Np5{UP=6Af3%03|lwEp+CUxkKyw1z_qF+Xf*GeI*#h}1_u~f=qDTw'
    'yE#hct!``HY7j5oL0|#m$6K?0GT&V$vqA3$spv0KNgK<vb*FD)<C#teDV+tvUBjC|2Taf-|I#=ZEFnOv2JhIUo|cW*fHOS|y3{xf'
    'lpzM3(g{eXNk<|)AphC1me|$yTH~(MZMqPfO#ukB$F1uc8XH7$;b3#VRGh`IuGE@}$cWV$2K3mMk#Jlp#OPDWQlwFHyzEM*3y73h'
    'SP2k8Dy1JD`~?5#Swa`LzDg&(IkI^{c2^Oba$-AD2x@lOwT&f_NZQ)ZbW=PiTJ!>JV}ZL4FQO|B#;5Jk1X&i0^RpMvKv*-V-v{>p'
    'Dr4)lQP?e_*f&Jb0`jnsDJ}ac^2KqVE7;O(Xk}^W;1<GZ>|!SmT{)MahQTK9-6>VD$&Og&#`eNRJK|~8;0;Y!XP@nIXw6W5&|n?{'
    'Ywap8DMxeIB5>h)yFdKYfc$uvDDveD&WuyUDI}S(X9LTHgd-b3;LP^y>74VS&02#YXFisJAs8PWF37Gh>V|J7oL&-a`buAMd<?Cf'
    'w)B$*;&++1{x?!rQ%a1IeJkpyYCC;B5tOEWp2*`&Cl$FGZLkr{Pof7hgmQ=a$;5qWAk}fYfapFYE>ztI)CoXAZ7*Ue2t!pzLm4*$'
    '=j&I#pOym+^C8C<wcCu0&79o1u;s40x04KpVg2X5Iw3R*23e{W#^xa-D_#Ng@;IpfT(4DZaULbEZ%G;O^3{{))$d;#Rj|{S3L5`Y'
    'ukGUFOfsq|d>X+#R1LVNFEqn28`HDlccx7?;7QX;o1<2L&_NPo*wZ(g@L;(Zgdyz$(+Lp`nPFT@;w;gtD0uuhILHg)oD)75*#wI?'
    '1eUj@W9Nk7#uAB$9>I#usOfq7W;Xim<YuBd-OgX>Szu}i4kz@6p%1{-nhgi$6v0$}(7C{Y5Xj=FF)88Q?qbM_rtDNAx8`7e)5y}s'
    '?Owq>LHnN*yBu8A9&ln@s9XXG$m0~^^yqwyLrzV{sX`o&4Fz)yCd=tKhZjUDuF0s(_n_k=!oDXAm?ov&y5m^%+Q?3dEbs~2&o~Ex'
    'jZJ`$@84&9_|>suVU7mI=vuKd6G4OVA~~`F7VT-43G%cR9G$x^a{F3wU{CVr^;piUiKT><W1d1OGsO`N9@K2m<}~7VIGQK0l2e+o'
    '!pt%A_VhiOhfBF0M~{$ttaE1CCQ25(*7NdTN6n;?+Rju0EIN6&@1IxG&QRj#=EQ#5@w6;>-(ADFY#7R=z)8ERGUln>I)RE5BWg}K'
    'QrAjjvzFXOPRkl8(X0d|ir?Z+-7U6&qpcUsZ;^LaYR&L6EI9+*7CKPw;1~roOD>%~rc<)srO>V+HWY8c{3=q<$XlaxSk%r#QQ}I;'
    'H>RngsB}*L=@$40xkxy0(rw)cl}l*VCl$Rj?HL$29{EMrJU0$c6*!Eo3<^i%w~X>-R?YDYQ*WE$gs2@2FDda?;dSFAW-C8j(5La)'
    '^3-RJl}7P^d48sb9;3TP65ELny!dO2vwVNTx^Jcd_@+T3P>$A$hnn(eCIwM%a2=RfcCORqV$5E}+85t(srAGhPsY3M1rQ3e5=Y|Y'
    '$QB7zBD=iB3)s<m4sNrey`>Y~&}=3e&W+ZypN!SRf0wXaAO2*-M@+_(`D9EKgQ{nZLUcMnuHA{o^Shd8JI3rl`6!*C)WxE)me%wS'
    'qMPo)nXj6i-sI9l+S*aJAencT>bdEb?l-AP5IXe=Khbp%@F~Ot&Cco?4Xr~Tn(2s*0%}KDJEobMjVY*85h=oVM7V)`?J#v9By;e-'
    'v;FSDcNJm*cH#&>qY?`ZE=hbqmnk9SLPlf&`cCHyq<N(MDBvIqCSP!(ZoUa<T>UJ`ts<mgIKYxhR8{ux!=2|qn%Ey2-dZ&6<0SG$'
    'QOw8nRKX;HH4JXRFgPa9V3e?nHBs!lcu@ay!v>F9YyzqJqOwpOIQza>j1$8<KI1-O*sFV(K|CFuJlK*aUco)JD4O{lx%wfKP@rkq'
    's!`FSNW(j%i0@D;xPzyWA3TUt(X-5B3`~RgmB>V&7ec;1b8n|yC7Zv#Q0raE&BT=@Tr-*|L0I`Ud@M_tQQ)+7fzO%`#+_kvDCcBs'
    'e(>es6toM)5hG}!=-<asp7O7qW4D{Ux~<Jxe=r+)!yD_r>lp!H4sB_DMe=&Nz}CqpzJS~S{~4DH3-l7@6{3GjEbMh%5#{`LBgzSz'
    '#?e0#{+0fH?1!%OOqjBG6^m(!&IjFAf=Qkj&96Tgf=MKUr0tuRdkNTWIIngVqxWr5NgOj2?QmsVk6+oLts2)}>^l1?;Cbx`2$qmv'
    'MVdhdRdY>9I9oO&F~T&^?Vq?fWCUTELC3=J@tsGP8*>hP9kw+}ZH?F*>n$z9nhd8#nWhI*1-=e*{?A1cC%2uM#^A+wKYVKgJ{>|Q'
    'eNaQVjs04!44x16oAcmZUtQwy6v<N1gpuqdlm<_j%d$m6EXS@yPL%=;*Dp*(V5w;~3T;it4s9F#KFEyGh_dP=B~^U0QSLKWF$bS^'
    '$axo8q|057TGPI!%Vl$x2<f)*D;LX64tujLNbF+3zjWTY_-aC-g<SeE7cB{m@A{yCNRZ+?(lXD>T9)MF?aPi(wGZqD(zUBjXPd^a'
    'qofZ<V%4LAMakk|b4eW3HWhM|CF7-L+caO%Q=tRRCkgrFry=Ft=G6`9w_jBDRX-;E71`YklE>wd^QP7n_D}~+9v#tT+AerU2a0WQ'
    'bhQ#y;BnBKG5vh?<X=y55$CrGWPbnsSYx-gk4nEW;sWNU0aR=U1WERHc|A5m!&1{gDjp(SS$Ah=T)tZ8RpLha4t`Ugp$??dG_KrR'
    '@Xgfg(Q1!t#&v$+yTM$EcnLY7vANnBWy?e3f`Kc6uC;lAlrYWV5m!ucQ9NzP=H}f~3>E$DkEf@c*~#Mj5>H&T*6bNnQAP@U23B;8'
    'EU!Dm{0s`E1atOW%;{cOm`BB~${vRg`z&PY*zxvqYQpb&t~Ibn26fV25tDWh;`|+qGrsFFwk;_Wc6t`eYDw=da*?Y*%5pH9gXthN'
    '3g+H1ptTxEHg7lOM2HwFpzH(&-$a@}a9KvGER9T<_h!dHfUa2KX0zZlO2smRn5W6y-5G7bVhQblkxDOG(_nJ<29sVnE!oE4_CPc?'
    'o1NjW;Rrg+c9MIU7*q04X^GbGsztmXIK4rS801E7X=_ngXM;5GNh%*g+vt0ctS{qB6B)!9o6)ab%*I|EgVZaT*pdmCOq`NQ33CGJ'
    'g2RwDRL~&AmdKJE<a0tpVMf-$^T;m6vgZB*Tt6sEVxbuel?CwQ7!dg#)boGBGo@z+Jv5#Vo16j1U<Ggt#$^XuMFY*|bTaLPRQW#4'
    '7#7fu$XIYI#EV=s92zSPhU@Y*MhofPVRGBxyIx|_)=_y0=PK(}c#7y0|E{Ru++jTf8igtz<!`Wco^rJ}7+SN7KwzkoD!tcwrT1E|'
    'yk}ZcPPT^y!zS|Ptrr8x3pKnERm`O4YwG6{SS19f_0lsf;R1WT6m!YZUmn}pE<J0^3(`@mahCh?GFWOYG8nhlAKBU}HnzQrk8R(='
    '$U2j#`LNR#I6wo`zM8Z-pyGcoZL*W8^;oqR%aCB(0oK1voPwL`<>`w&7r-r@P_-O^Cw;L5nExa-A;$GNq7B^k%nIn<roM?dpH_#{'
    '99^4lE55xg8Q888{I|C`#P|{(z(dW*K;7U;uZ6g-@l)a7BxXmj-1A^IAIN7|%kEIRp@wljQw`YDWZt<jminjl;QilgAC9(yMnl}$'
    '9rGZ}%4w0+yAS=Oho4HF5b&o=`;sHFY8<D^aHYm7%@4k(R%JJ0)u3)yEU4q;P^`V9x59iU*YjD+UNPZE7XA&!M3YYk(}ZI-gFPP('
    'jjUS}zx?M>u;AFLA~HV|lOvX&ii{n~zRm(er@qXs3~8$vO7P8vi%+Dt1HykZxPPfC!o=U>SdlWVOl-i2$prRnO{Z*y*SW++#{y)`'
    'lB4crOTjv8?yi?y!AU%k(WFs{sUeE3p^$T1LPg&hSzIX<yVeJ-;`_0g9SvzT$!suUx5rF}-W|~D&7pC(sr9UGN2KN=!RF*PZV7*?'
    'G7gnWcdUrM1NB{UR1`Dk_#qXbv!|!X)Qu}c%HT;Cl(FDR^hI)%k4~Yb^K`jdy27EtUVGe1fjFOvg@e%NS#hgO7tQQ0fzDtA6$&Dv'
    '&SnW4d*~|Du|XI?>ywC;I@sTBBlx0*7z{d4Hsma38VdDs(2frz#nR3mod-fngUe)ylh*~|e?yH%M$y5vd&e%geGR)xe}Qvs1fQDe'
    'BxOOG;x8+}Suej?E4c|AJ<=^KN{$`5Mapw$`OX^2&K(6iIdyZHFlVJK=hKk`msnf6%smj4cS+=j$?q^T-$`7JOI}5Dm@^R(o$U&P'
    '40EEzd)e4rtjg=LQt?wRG<{sui1;4B_$F~C@&#EVZ<W6#c_{pxow+tS<xa8Df30q{JZzhzbN`gnBYK)USOt0x>Y>IpSIx664CT2f'
    'S%Cs3JyoL;YKY2oKWt`$fWPL<W;&k|iQP(Rohz)nMNsF+=?H%FlDbt=x-tpfEdshS;oJtYxlAti$%O0i<Z++!Xs#r@05G<yXRl_H'
    'xtIs@K02u{jRKw}aJkz|dz7xZTSk$@?KZ8axu8mUPqdG33Tv-_I3{+J&TxJ+O+qf$Xe?Vpn~jduMQz>VTaqY=x;p@S%HyK7x?HKz'
    'ljXnkvVq=BJNJimbDLT978oBpa?Tx(6x+=2*w=M=T^e;Xg~~Ja^>PK9ptnV9pc+mhcU6;U@}3<KU9{E+XxYhG0MQ{`Zv<4ky~t|K'
    'JKkx2UI&{I#O%;bm}2dje*>c?RN{lc$tzmO9Gl6xjoHj>(p_|5Zh(N#Dg$C8J_(CNoQK-)d@3R3R$dp3Kk0mmwMjP%R}Lc-9D~$S'
    '_;v(1vHjU(F(oQ#EGZP($G{*^lg%)TMN+DFJUK48lva38rwMHG$J9lY4k-tOuc|9fb-_N=>OI}q8gJ!hd-nFlcjpc-<|2nMop&1z'
    'nEdr;t)X>af<sj$h)=?3Rho~L!yAn6IJ{?{JG{Cxyj#f+p8)w|SAU5cz4o*5*lW(%H#B;?Wd?U5Gz|02ZgihCan>kd9<{`??&RNN'
    '_L>tDseG^$LL1$|Xpf0EsDHYxb56qtPcr^dv1}HLTBqHiPRh8*0ci_121SB|3L=tax2P;>p@V)$neqodO_N?|Jj1Y)C+v1wtM6A('
    'a=fA>H*AX*df=Oc4dT9${u<nvJ508>=yW96afN^Olo;arwbH_IRyv)!zhk*JEy`m9$2YHE5UON*O!;RQZw3rZD%Zi4jdb+9N8$Dl'
    '$!z@XIP$j1rey#PXjmR`W29~26j!h-bR(y50@pCkxth?ivyK>*V9zxqkDSR$-!hWma{PXueZULOlMdk4Pb$H>@A1j)Caiy?{k@Y<'
    'N2@0g>kMk^8+mxb*0bx1JX+UJI^C?Ff85?r9Ix+a!x_bU8+7&X<WuG>iDR3^do(2BNX(K(#*;cTdI7@tmLTptnLnAU6@Ew{Gx|`m'
    '-|UB1t?TDEgsKLebLMhhF_Ux70jY30E#q3=X#76@Q(E2oBdxAjqftI9RC(_NGJj~LR<|4fv4%gbS+i1TV$6N_4tuxk4w2y{jlQ0Y'
    '&n7dLUOy9bh6bMuA?WHlffP{jO{cS%61Kgekv!=D)jfqEGd)XYuNK4kfCxjv#OY)f-e&|~0mj?QtT5%GQK#}eJl(bNxDptM`xBZ^'
    'iC6$g{b|j|6+u3KQU|>im`Y(wd*r9A!=4-6sGBUE?!Qas&pC`-c%MGF1C!U^xv+MhZvBe+h}S&AGc&?xuywBlPCH8B_&F$@%fH~E'
    '1?QDCpTtKz$@Efh1ex1dW^2fjat>7}F$Eps3+(mOV(*XEtHM!Yp|*`RX9LYue+^FDQlX`mgbr#5wFCB()+GwGX+sq`x_7p<5%(?r'
    'WG<<MBa7H(LnUlhZ!q;qCy$!@CO~oyGn;RSPH7-KSawNiqY!9yhAuPcuB#Y1A<g;3ajskGZ2Run_dmY=LDO&<mpsI3WUp%R#A}Hp'
    '4hu$`WqZ}lbDaOu;h-xX)(}^u4HHE`WD;?v*(mA@FX6rIi9whX84bpdZK0FU**RNM8?OZ3$4q$%LBd(Ii6;?)1FdVsv9R@uS+7=('
    '074uxCKDM@0OIra^n-V99M6P`8}Oi^F2>@jPbX?u{pZ!0COmRt@C6)FV33GK2_dIp%6hh_9C(avu#t<<<ecBxv`^p&fpY~;L=Tb|'
    '8r#h#$sx*l(qCcLrIQZKs-W~Yf>m?na2^PmuUs<!#IHUS5<);Y4n$a5(4yQZp8!cSqJ7}7<S<S}O$%u|Jx`*m$%4w7d}KIrPRv2V'
    '8J8RmnQ({BQ}KZ?BD@}U2`~d$hGV@Jp<VDAA&M%i2?FrQH3YYn`O?c*PNz;~DzUn9Tv|K!suw!HGme$k6`@+fFo#(k(BRHX{W=J6'
    'fB)TK#l;<6edaj_C7}NSf<kBtuMx8m!qEpQoe^9~VjCKsJwI(oaxLCy#19Y+nkWL)|EiM!r8$Ys__&E{Ws-VlP%b)_@u9_TGG!Ot'
    '$*{^`qtbE6(tsp0?^;~Grwo_RCga|q--OX5B;>zYHCL}+RlT!npq&q_pq4KXX=GPgfpDOcR7SqI={?b4R|(fqGJ1Lg=c{8PIF1|x'
    'TQ`CM?vnFYQC3?Q`&d`~C_hI_dVuJe5uxj)vCDZfI7d2=^5fQrDr+iBT2!L=Idt(*2p<({E?dKD$CiSMD6k6f)x~aX<mPpye)UT2'
    '?2KDQ7Y-M8H<-|w1H~erI(K2Xd`Q#vm6Tn$Ww6*uw3<rrx0wST<HhKSezCQ(vH!ql;l*`$pUPJFXjFoaQHf0fK0SW?Sdi|WffiJ&'
    'K-kI2N%+w!dL_t~3YVV!NW>%2vq8B1Vhl<7J3852$eEt~I42mH%wDyo+eBh#PtRNP);p>Wn%D#rtFV-6+XPsa4=`=m-pE6_q?-4N'
    '<CT@Za>n%%E~l(U*sM>d>bZon@|ruX6PS!900{5>w=l$b^X4hch_I$~<lk%T&dF6123zTefbF)HgXBu6c#ii2Iq9_+asfqH^Dd-#'
    'V5ITGPZA@bl6}({Ti7uc1I`xXa6XxJE*gz@JRl%Rf(7rDQK#kKOVlC4h4~McO&OrGY<_{I7RbHMOvQ>VYRJWWJMwyEM}zUQ*_%Q4'
    '<-eaS=J{8WDZb?GaCP(I-(9ihPEWI+r`gZ%KRUm$yyoF7`zfTg+mrb+o8V}er!1P7d4p5ze5V<7Ckt5H@zT)t^y_Z3Oyp0d0K?Ys'
    'u{WKQpxf?eO?6VY*<sn(=A%|lMlZq5IpFAKp&GXdm0T{X2;0@1&Sys7;V5yBEdbh5&^MUa4AZKGL!fl?+bfH~X5|$;R|S4N<XDDU'
    'sAOQvO{y<fR`<fCWe+RnrhpI1P=00QcmcDLOIvbzOG$Gm*mXsIt}F|ykCKVzw~&gJOjV8)!kzJtm7Q>gZZ0)kS4LvW%a0Yy_0%YF'
    'o*o4pJ5nFN|9GTfnazvK<&iL9@+_?8yEkv2eJ?Sg!!!230V{SE@E@q7BMq^!m9dS?*0hPX3Z}4r)5ruGf!b}*Kos5P$uNXwJH&t_'
    'y5-{_j89Kt5W9!$=vx9}92kZ>=ndFtXD<%gbo#f<e1{o5uf{ZCj7Hd8m>NBUB@G_5e8y<%yd40<Y{J;CxA6K}OrEe}Ku6{+ejmAU'
    'tMt8jJ~BjxzQ^w)$-r((cC<SiBkDMdm<VRL&h|a|AXX6b*w|4ad?AKIUOP@T8=6}`asz7df)~E9g~y_>SNR@kNAj=i9w3X!F;Hjq'
    'qLnt;7$(FB3x~!sXmntx`0hywisHyuG@GdsfE=($7ps%X;tjLa8yYx}AkO+82$gqp*KVh!C9JWig}uV$L$~<-z<cx#foWp>8KBY4'
    '+AG0y){fY+%XXm<J8K&Z)a&gbmTlquI7g<=>DZU+3UE$hHs=wxYKXv)O2ecLW7tZar5ZEKT@l4!B@Odo%_7{*ClhJ8<oqM(JFof!'
    '=xYdy_G|w*pxQ=qh{v-0>_|jy!#<iwD7MRzj<Ef!h6*TIH%P|hrIlGym0^q0GUx#}#seoHW}!Q}+AU>*ePu)+W$(C<H5G7!#97)$'
    'IhS|=`Jw$z)R%)5%u1`LK+)cS`K(3*UmA^mGJm$1&64r_*^kekNXd-A5V?sOSrgI2Cc06U2*VM|5x>bH{!FO@EoMT$IYJW2T+e2f'
    'nuUbG4`$=UNVSNEvZBOO6#+`I$1xjg2p4BeClo<cSXLu}e8Ko>jl&IU5F7XPy?ghbot-t$-y*W+Y-?-l-e15a-201|gnNG~k8tm='
    'ViE5BDICJRFTfz&E9DQ4?g5VO-D|#j^PHgSHF2i*nlE4f)cp4Ci+{4n?OyZ0&h^LZA73@kU+L4mdypsy6m&xchRfup!U+YU2KPse'
    'L(@yQodzZ*<Ot1|q;<l9n<lflk<+ImI4n?^!4grZc~)r^14LVVcN-?6N+=<u-lBUdN|LXGl&8?JlQoqvrZBRvxsCT!ALbKW=AL7D'
    'FE8%h`&)1ZYik4vHb`KyOlEXY5gyz4RcM&QUl!ed@^u<~U$4~~R5fhF+odAz7L`DCLH6cI+2dh4Y>kO}OorXGdawEJ$JZ~g3%~JT'
    '!o7MWIIINk*sKi!pSy$3+!8;>MD~n3$8m+VjK_3;NUfL!nBS^Zhpp8)J~Sn>xTDAQhjm-;fGe)*?NEB7(%5?0d*0D--s&H3Z86&K'
    'r^iGj$?E?t_;zq@ww=wc{!weOVzbKMB|s_wxVo4RhH14koi-8QoJEzvU^Jb~pb=|zo09bFp1VcP`_S$x%-@cCTa(L9ll6<s`aceT'
    'qv3r**IU^MKu_spG3zAhv0_x`Bi%UJSGsQe&piSIc4j$#zeSjL>qGX5{n46SzaxwBerxy59xct*Za-mp+i{ELd#p}K7*&Pe%HlX9'
    'EcwsY?rw5Do$t<)VT;yv*P0tr``g@|GZcc-_9RV?1q}FgWOi*dLCn)ow@Ew$1Dt<zlo%HjPLg5>-So)a46dxL8+?s;ZSz(|PdY%3'
    'q5k;$7f+rypTB&C`?j88iMNw87{tM=DG7Q^W{o=>HZBfWcQtSQ9M%W)`y!#8CjulAlQ+oQ(K4e6ZVkhb<DLzE(33VA-U@^DiL13h'
    'qJ7R=!8Xx=0+KBbtOmPtMBvxKR}e-{CUh+SL{+AbgkN+@Mp1@qaMc(4m2Qw^FAZ(;e%tqQ(Ou@IvufrOPaEOwgrZ}M!YfL_$pks;'
    '?<~~z?WRPEv&|k_@|8v=!cMb`Am3WZf_4nx$hPkc-e@U8zz5%+5ko-tFro;zXqIUZaw@@gGQUa^aY{@5{A^A(wozp9JzU3NcJ2<C'
    't}Y!@<P1huinBM-WmUKu=z~IYz*qkJ+6RX~mgZ>d3}j||HCFEvvkL$9=IsyXXHTBJXuf*+y7~6af1PK0#Jz{!t~zbf3PJ{dEZ=!Y'
    '_6o3+KmJ~0U%CYrs^{xx)1h@V*W*>{LKr<Ml+Y43(@|<Rr%?)rN;2f&ZrB>NiLTf20b^p*OtQy}Dlj#wBN*W{t({O)@?x5i$Q2c`'
    ')dgL+iwQxx1<iAj3jj}K?3)q3H!>rSxZ+ufpgbD1;Ap(^2p*^?JdC{l>|n%|0n6QKO|xPAm*qmGKzYwpWle|LrtJ!-a5kBYLdm0$'
    'b@omj#f*`#<R?Y=jFj_H4{s!mmr3EMi5T)-Yr6qL1V4G*1TepS^X3&cH3I(j-#&lw+6)Y2P54w3XH`H-D|sAP5myDovj;lVxv0#P'
    'Dncp)lcB+icR0Ih`CiiDt|&E)0{WUMHbmXjiAnU4qbv~pqzAJlZkMRJ1;hjwv&o`=5j;PC_2N|!LZo9$bCzytgt*3W$BCaW`)b!P'
    'm7q1bQ^KEgStYv*eKd%QQ;CXX59Dx6*y$-m5wD*7tNHZFyJz2LsEMd6NYywi2mFU<lC7Z=$xb4Osy}-!?C94T(jQ>XD(hC7s@__z'
    '()`8aQ-TVRItflOB6kSZp6Su_7}oQhA(NUQ8Vx>&81Y*D^YZ3Ou5r;>#CTguFstl+l|^J5t7CIMLfgFfCpCX~@!#hvS$vSaF?7pE'
    'RJDoziW}gr^qa95du8T%zB#>sSUrPHTF1~rjJRDB*+TbJ)g?&8H^iq5aO33AS>cqctyvn~D~SN-Z`D{grB>V&2TvV3c)w%-*+%=l'
    '{(&C%BjrVE_L5eoG{8OY&f}oI?}2-AWdYc7Q-z(dd1E+H##C)GB>xHGDhw0<z}{5<ikZj$EDzmT#s}^!H6ld?p&p#myzwko8J#*^'
    'gjo`5n+Te7;g}rKb5(612~8^tu%G@lOJ&*FRZDhqylTyI!92?rt<`2nX@mIqtkQKsGhDfJYo~2)$U)BzuD&lAi|~5%XRJrokGlx3'
    '{JC4~GoFZ@>Wp<%`H82K{3@kW_H2d;t>x7Jn$z*O;M_Bpl~gFkw5t5dKMLR&pT!Xo0a7yN>zl*In2Y~!*^;!_=uB*=L3kluicFN1'
    'Iq?Di9u7tW=Zbbvl-b=905umo48+oGdrVs)8rtSmK#2Hrz~l1wq~Q@JhrDa?L;e^OUv>UUTv^k3OG&tUnX5iqU+-rpWn12G@7ZAJ'
    '-Gc`-FMc8Fs@QL2n_r*k4a*_fxaAdX40^>%kSqhc?Kl4No5hn%_)VB$dM!atil+Fz+ZdkrGh$o)CZQFqK+?(_aD~Mw`g7VH>aXo>'
    'BPUeaR_qs1_x)CfjT*N;@CPE25h#<szu#hk;T9ZS@Wg0MOI>`d1Q!rIcZqB^#X(tqWa=QbmGJIznR{s@CiVpj>7jzu=0!Ovqod#Y'
    'R%Nuu0b79Z1HU(klrNydd7waG71G>+JZU?gNriNhz0xvb$RO+>M?eq3J`mq<fdrG?$md;+X+x>Ngqerqbv%fxRR?>){$v?9BGlz#'
    '%r=1{lVy>DjtF`rC0-+w8rULdsfHs1QqllHb_mQHOcq5=y<$n71_LmDdqCxqkz1W~3;IXfG<r7=@9`C}6=R>yM2qjYa0<3KSoXbK'
    'u;ML({w<z^({iIAo|I-#_CrRA>q|X!^aV!!-gu&tu{h<A^%W0&+wG)en0FW@H*Jo$(<+X+Bdz0h&1ecMdwZ8&cQJ<Ptu|jDxl!1;'
    '+oAZ{)JmXO3V-bSxYgV)D$e><0&k+(*z{I>R_37;;uckoSnkP4U6M>-1!jkYp!S6f6^+3J(H=6^X*afMp)RnHkV+xw7mUd0s{7%o'
    'XasN_p1(TMaBNTk7%v(TV%U~?EhY@G4RV84&0)eSG`QPhC?XBT#(A|&Y?aWlS)V1loQ%;<y>sJejYQK7AtV-(vBDN!Zi!?vqXg~+'
    '8~(@fl@Yx9SV)e%+`tHwePQyt0o81TCjvs`owF7)Y|?~=){as{t#))zHyz<3O`4)+XG{tGwE_W@?!<UqrjC=qE_<lA2Xj_}HSjBd'
    '3w;?SUXiZ{$X50IRJI#!<`;n8Vg6AR{EmQcv_y{|2hZNTIcq+B`Ho+rypEH@98z=q&4nOC!yLP~Q;(rI{43d{T4&ZX2i6eI5u#^H'
    '2kL_}C~pV<WWm{a^X$dj=Gl{%Z>@}dW6&wT+|DX1aECAwx=Egm2VdFgNEGpma)kg<jeWD%>d?K*A5PzVSAQY;LJ&S!Bdbq}s<1?0'
    '+XK1fjm-fNkCHyd+8}$Sb_UstW%j}`CMa1IQ(uRj)weG)WfV^cV&x|5LAbOLMzbY>w~Hm68AAa>l_))SmMuJE`a(Z~?rX(O!G@G-'
    '__O&zn{V+xl=UuUOQW{p@5L0lQfT3OeI+<8m6)gXjA%DU83J=z)f|)j1rk$`+jbeD>DO@H|D909ah|@gg_Xnw4nP0#?Cj0kcQ2kd'
    '|Jgi!@%lU~q*Spd2j#C~0*YTn8_@FpOjFxB{oG(57}GEG^Y+C*e|&NNj?d<~Wdde|@2)FBGEI-;WcQ&iHbk1&MmguMGJL^=WSylS'
    'XY=s7I-OjF^{6^Zh&bJaA2so>v;oIZyTdw?TFyEnI6`mJiR53kT8%FEC$#HsEc@!14HtB<*Hxpx>&hhW$+Cbap~lU~-&d>E3hT+R'
    'M5YnpUoeD}ec-5Cl^(;Wx)`T)^Cz%mtVb38Kt=K&<l0)o24d3XkxM_dMgva$`V`~IT{>Ad;&VHf;d{=5wvAEH9;20Q45kw3dkkzx'
    '-%}ZCbvKUKww0(09)qqZ96{j*%3PeIy<{24+08rjDVgp-=<ou|Wd%C`Ln3rRwtxccu-rG4BxNFn67{r(&rCC2=1TrgjYQOy^-hc+'
    '8}WU4Ua*OUz2v`3{>w=)J@&B5s4RRh%o&d;|AW8CMhI6nNi*)x!~><4;58F^iy-A0j&a&Bo+{B|*bBd)py4jvJM&5CGlr3O7%xp*'
    'm+RW%RlU8jG9;hT2Q(n*4M+E*9^FEDIxJG+a%G8oPHP`a#zbFA-l}$K6{}qZ*{<^tyVc|3#k_J$GRKVE%`G*6&2$mPTq{?6xxTnH'
    '56nCzZoA->tbE7G|E@R<`jF$esWGmYq?t7FE$1XlGRxUOGwF@0?gZcS(tKc}W;)Q?zXKS&+K0_#f3(*ddy{plE^mi5$9817Nhg_z'
    'M6FYN83t3C{lp4IK2}|UMu~)_^fcWfY4IBcmTY6aWjx%VV|BxZ6~_XQ@7o`$ondP<MMU`HTJ&C3hyBD1oKX0j2nPJLEodM|vomP$'
    'gI%_5vp24_9|yo)QGA&%K1c_0R_5<TaMxfugU%!k_kBYzLs#13iMM>d-?Vn$Ywjf98r@(;8JRk*W;--M)WL9Tc-ik((#!7d3H?fm'
    'tNDm^<OAdZ6X9E0%^X}+B*26vcQzpLqugldZ*|m4FOzN)9+2UGbdqK^=5OOeMPgQmTf2jCZ;S9~I_V*vU}$at1wu(HRG4LIXW8B2'
    '@GlkK%*Y-gA4(mNX3UOY)t1&u2mR4x&<$~MnYD;Y(3;m+Mpm!BZCDgphIS2k0ok@sTAZ_X%|kq2r@>yG{TT=Xl52=In36~Xv7!}y'
    'J;IH!ykf=-aFTa|fYyE<><8aHEu}KP7AoUwD_(6T!-Opou<>8>)V?-w@HLYd*`q^EWG@Zn@U_oJgwh?+1z)>XRWDyZfAO#8>C0Cy'
    '-!c6yrLxOtZQ(E*ulF&EEG_TMAS`Ly!HHbjQaN8EoU1qI9fZiwjY9efy>efH`+-cr?VXUt!M@pl6V7JLBWeUM$5Rl2u=5pw5j*XM'
    'Yi-~#Z$5PxvCIY~@=)cPS@{I5XMyGDD`<fK2f-tZNd'
)

# ---- reference.py input generation (trimmed copy; scheduler_metadata is None: the kernel never reads it) ----
HEAD_DIM = 64
HEADS = {"tp8": (16, 2), "dp": (128, 16)}
SLIDING_WINDOW = 2048
ROPE_THETA = 500_000.0
QK_SCALE_FACTOR = 5.4730064863875
SM_SCALE = HEAD_DIM ** -0.5
RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
K_SCALE = V_SCALE = 1.0
SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}
COHERENT_HEAD = 2048
COHERENT_ALPHA = 1.0
AGENTIC_SPREAD = 0.075
CACHE_CHUNK = 1 << 20


def _quant(x, scale):
    return (x.float() / scale).to(FP8)


def _dequant(x, scale):
    return x.float() * scale


def _rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, pos, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM))
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _longest(ctx, spread):
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return math.ceil(ctx * (1 + AGENTIC_SPREAD))
    return 4 * ctx


def _lengths(gen, batch, ctx, spread, device):
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    u = torch.rand((batch,), generator=gen, device=device)
    return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))


def _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, scale):
    hkv, hq = v_cache.shape[-2], q.shape[-2]
    group = hq // hkv
    tail_gen = torch.Generator(device=v_cache.device).manual_seed(seed + 0xC0DE)
    u = torch.randn((hkv, HEAD_DIM), generator=tail_gen, device=v_cache.device)
    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
    qn = _rms(q) * QK_SCALE_FACTOR
    if layer == "sliding":
        pos = seq_lens.long()[:, None] + torch.arange(q.shape[1], device=q.device)
        qn = _rope(qn, pos, ROPE_THETA)
    k_flat = k_cache.view(-1, hkv, HEAD_DIM)
    v_flat = v_cache.view(-1, hkv, HEAD_DIM)
    for b in range(len(seq_lens)):
        seq = int(seq_lens[b])
        first = max(0, seq - SLIDING_WINDOW + 1) if layer == "sliding" else 0
        past = torch.arange(first, seq, device=q.device)
        rows = page_table[b].long()[past // page] * page + past % page
        keys = _dequant(k_flat[rows], scale)
        logits = torch.einsum("lkgd,skd->kgls", qn[b].view(-1, hkv, group, HEAD_DIM), keys)
        best = logits.flatten(1, 2).amax(1)
        if best.shape[1] <= COHERENT_HEAD:
            continue
        tail = best < best.topk(COHERENT_HEAD, -1).values[:, -1:]
        vals = _dequant(v_flat[rows], scale) + COHERENT_ALPHA * tail.T[..., None] * u
        v_flat[rows] = _quant(vals, scale)


def _fill_cache(gen, shape, device, scale, seed, dist, layer):
    hkv = shape[-2]
    k_cache = torch.empty(shape, dtype=FP8, device=device)
    v_cache = torch.empty(shape, dtype=FP8, device=device)
    mix = None
    if dist in ("soft", "coherent"):
        soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)
        u = torch.randn((hkv, HEAD_DIM), generator=soft, device=device)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        lo, hi = SOFT_C2[layer]
        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)
        mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, hkv, HEAD_DIM)
        for at in range(0, flat.shape[0], CACHE_CHUNK):
            n = min(CACHE_CHUNK, flat.shape[0] - at)
            x = torch.randn((n, hkv, HEAD_DIM), generator=gen, device=device)
            if keys and mix is not None:
                x = mix[0] + mix[1] * x
            flat[at:at + n] = _quant(x, scale)
    return k_cache, v_cache


def generate_input(batch, ctx, heads, layer, seed, q_len=1, spread="fixed", dist="normal", page=1,
                   kv_scale=K_SCALE, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    hq, hkv = HEADS[heads]
    seq_lens = _lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page
    longest = _longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    shape = (total, page, hkv, HEAD_DIM)
    k_cache, v_cache = _fill_cache(gen, shape, device, kv_scale, seed, dist, layer)
    q = (2 * torch.randn((batch, q_len, hq, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)
    if dist == "coherent":
        _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, kv_scale)
    k_new = (2 * torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)
    v_new = torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    gate = torch.randn((batch, q_len, hq * HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k_new.flatten(2), v_new.flatten(2), gate], -1)
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=device)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": cache_loc, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q, "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "config": {
            "batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": HEAD_DIM,
            "page_size": page, "window": SLIDING_WINDOW if layer == "sliding" else 0,
            "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,
            "qk_scale_factor": QK_SCALE_FACTOR, "sm_scale": SM_SCALE, "kv_scale": kv_scale,
        },
    }


# name, batch, q_len, layer, spread, dist
CASES = [
    ("d1", 1, 1, "sliding", "fixed", "soft"),
    ("d16", 16, 1, "sliding", "agentic", "soft"),
    ("d32", 32, 1, "sliding", "agentic", "soft"),
    ("v1", 1, 8, "sliding", "fixed", "soft"),
    ("v16", 16, 8, "sliding", "agentic", "soft"),
    ("v32", 32, 8, "sliding", "agentic", "soft"),
    ("fd1", 1, 1, "full", "fixed", "coherent"),
    ("fv1", 1, 8, "full", "fixed", "coherent"),
]

T0 = time.time()


def over():
    return time.time() - T0 > BUDGET_S


def log(s):
    print(s, flush=True)


def incumbent_namespace():
    raw = zlib.decompress(base64.b85decode(INCUMBENT_B85))
    assert hashlib.sha256(raw).hexdigest() == INCUMBENT_SHA256
    ns = {"__name__": "_icdiag_incumbent"}
    exec(compile(raw, "<embedded attention_a3_scored_2.5690.py>", "exec"), ns)
    return ns


def capture(fn, data):
    for _ in range(2):
        fn(data)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(data)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    return g


def timed(g_replay, flush, pre):
    """Optionally flush, run `pre` (untimed), then time one replay in us."""
    if flush is not None:
        flush.zero_()
    if pre is not None:
        pre.replay()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    g_replay.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0


def run_case(fn, flush, name, batch, q_len, layer, spread, dist):
    t = time.time()
    A = generate_input(batch, CTX, "tp8", layer, 1000 + len(name) * 7 + batch, q_len, spread, dist, 1)
    B = generate_input(batch, CTX, "tp8", layer, 5000 + len(name) * 7 + batch, q_len, spread, dist, 1)
    torch.cuda.synchronize()
    gen_s = time.time() - t
    gA = capture(fn, A)
    gB = capture(fn, B)
    cold, warmi, warm = [], [], []
    for _ in range(REPS):
        cold.append(timed(gA, flush, None))
        warmi.append(timed(gA, flush, gB))
        gA.replay()
        warm.append(timed(gA, None, None))

    def st(v):
        return f"{statistics.median(v):.2f}/{min(v):.2f}"

    mc, mi, mw = statistics.median(cold), statistics.median(warmi), statistics.median(warm)
    log(f"IC case={name} cold={st(cold)} warmi={st(warmi)} warm={st(warm)} d_ic={mc - mi:.2f}"
        f" d_data={mi - mw:.2f} d_ic_min={min(cold) - min(warmi):.2f} gen={gen_s:.0f}s")
    del gA, gB, A, B


def run_all(dev):
    log(f"IC_INFO {torch.cuda.get_device_name(dev)} torch {torch.__version__} reps={REPS} us=med/min")
    t = time.time()
    ns = incumbent_namespace()
    fn = ns["kernel_fn"]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    flush.zero_()
    # build the extension up front (kernel_fn compiles lazily on first call)
    ns["_module"]()
    log(f"IC_BUILD {time.time() - t:.0f}s")
    for name, batch, q_len, layer, spread, dist in CASES:
        if over():
            log(f"IC_SKIP {name} (time)")
            continue
        try:
            run_case(fn, flush, name, batch, q_len, layer, spread, dist)
        except Exception as e:
            log(f"IC_ERR {name} {type(e).__name__}: {str(e)[-160:]!r}")
        torch.cuda.empty_cache()


def prepare(weights):
    global T0
    T0 = time.time()
    try:
        run_all(torch.device("cuda"))
    except Exception as e:
        log(f"IC_ERR top {type(e).__name__}: {str(e)[-200:]!r}")
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    log(f"IC_DONE prepare {time.time() - T0:.0f}s")
    return weights


def kernel_fn(data, weights):
    """Pass-through: correctly shaped/dtyped zeros (tests fail; only the IC_* log matters)."""
    t = data["config"]["tokens"]
    dev = data["stream"].device
    hidden = torch.zeros((t, HIDDEN), dtype=torch.bfloat16, device=dev)
    stream = torch.zeros((t, HIDDEN), dtype=torch.float32, device=dev)
    ids = torch.zeros((t, 8), dtype=torch.int32, device=dev)
    return hidden, stream, ids