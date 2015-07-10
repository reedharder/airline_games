cd('C:/Users/Reed/Desktop/vaze_competition_paper')
fid = fopen('coefsN.txt','r');

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
freq1_low = zeros(41,1);
for i=1:41
    if (freqs(i,1) <freqs(i,2))
        freq1_low(i) = 1;
    else
        freq1_low(i) = 0;
    end
end
[f1,f2];
[fval1, fval2];


%compare frequency coefficients
fid = fopen('coefsN0.txt','r');

fcoef = [];
tline = fgetl(fid);
while ischar(tline)
    coefs_cell = strsplit(tline);
    coefs=zeros(1,12);
    for j=1:12
        coefs(j)=str2double(coefs_cell(j));        
    end
    coef1=coefs(1:6);
    
    
    coef2=coefs(7:12);
    f1 = coef1(2);
    f2 = coef2(3);
    if (f1 >f2)
        one_bigger = 1;
    elseif (f1 ==f2)
        one_bigger = 2;
    else
        one_bigger = 0;
    end
    fcoef=[fcoef;f1,f2,one_bigger];
    tline = fgetl(fid);
end

fclose(fid);






% calculate best response analytically
fid = fopen('coefsN0.txt','r');

calced_freqs = [];
tline = fgetl(fid);
while ischar(tline)
    coefs_cell = strsplit(tline);
    coefs=zeros(1,12);
    for j=1:12
        coefs(j)=str2double(coefs_cell(j));        
    end
    coef1=coefs(1:6);
    coef2=coefs(7:12);
    b=coef1;
    b1p = coef2(2);
    d = (b1p - b(2));    
    f1 = -b(2)/(b(6)+2*b(4)) + d*b(6)/(4*b(4)^2 + b(6)^2);
    b=coef2;
    b1p = coef1(2);
    d = (b1p - b(2));    
    f2 = -b(2)/(b(6)+2*b(4)) + d*b(6)/(4*b(4)^2 + b(6)^2);
   
    calced_freqs = [calced_freqs;f1,f2];
   
    tline = fgetl(fid);
end

fclose(fid);


%check analytic and myopic solutions for equivalence
tests = zeros(41,1);
for i=1:41
    if (freqs(i,1) > freqs(i,2))
        t1 = 1;
    else
        t1 = 0;
    end
    if (calced_freqs(i,1) > calced_freqs(i,2))
        t2 = 1;
    else
        t2 = 0;
    end
    if (t1 == t2)
        tests(i)=0;
    else
        tests(i)=1;
    end
end


