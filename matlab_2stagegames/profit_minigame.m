function g=profit_minigame(f,markets,min_coefs,fprime,freq_ind)
    profit = 0;
    for i = 1:markets
        if (freq_ind{i}==1)
            vars= [1,f(i),fprime{i},f(i)^2,fprime{i}^2,f(i)*fprime{i}];
        elseif (freq_ind{i}==2)
            vars= [1,fprime{i},f(i),fprime{i}^2,f(i)^2,f(i)*fprime{i}];
        end
        profit = profit + min_coefs{i}*vars';
       
    end    
    
 
    %dot product of full coefficient vector and frequency vector to get
    %profit
    g=profit;