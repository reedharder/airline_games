cd('C:/Users/Reed/Desktop/vaze_competition_paper')
fid = fopen('coefsN0.txt','r');

%myopic best response
freqs = [];
fvals = [];
tline = fgetl(fid);
while ischar(tline)
    coefs_cell = strsplit(tline);
    coefs=zeros(1,12);
    for j=1:12
        coefs(j)=-str2double(coefs_cell(j));        
    end
    coef1=coefs(1:6);
    
    
    coef2=coefs(7:12);
    log_=[];

    f1_0=1;
    f2_0=1;

    eps=0.001;
    diff1=eps+1;
    diff2=eps+1;
    c=0;
    f1=f1_0; f2=f2_0;

    while (diff1>eps || diff2>eps)
        log_=[log_; f1,f2];
        fun1=@(f1)profit1_stage1_valid(coef1,f1,f2);
        [x1, fval1]=fmincon(fun1,f1,[],[],[],[],0,inf);
        diff1=abs(f1-x1);
        f1=x1;
        fun2=@(f2)profit2_stage1_valid(coef2,f1,f2);
        [x2, fval2]=fmincon(fun2,f2,[],[],[],[],0,inf);
        diff2=abs(f2-x2);
        f2=x2;
        c=c+1;
    end
   freqs=[freqs;f1,f2];
   fvals=[fvals; fval1,fval2];
   tline = fgetl(fid);
end



fclose(fid);