Cold = 10000;
Cnew = 431.19;
Mold = 1000;
Mnew = 242.36; 
base =[-95164.0447,-36238.3083,1148.0305];
empirical_freq=4.5;
transcoef =[-(Mnew/Mold)*base(1),(Mnew/Mold)*(Cold-base(2))-Cnew,-(Mnew/Mold)*base(3) ];
