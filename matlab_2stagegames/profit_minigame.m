function g=profit_minigame(f_i,transcoef,players)
        f=f_i;
        if (players == 1)
           vars=[1,f(1),f(1)^2];           
        elseif (players == 2)
           vars=[1,f(1),f(2),f(1)^2,f(2)^2,f(1)*f(2)];  
        elseif (players == 3)
           vars=[1,f(1),f(2),f(3),f(1)^2,f(2)^2,f(3)^2,f(1)*f(2),f(1)*f(3),f(2)*f(3)];   
        elseif (players == 4)
           vars=[1,f(1),f(2),f(3),f(4),f(1)^2,f(2)^2,f(3)^2,f(4)^2,f(1)*f(2),f(1)*f(3),f(1)*f(4),f(2)*f(3),f(2)*f(4),f(3)*f(4)]; 
        end         
    
    %dot product of full coefficient vector and frequency vector to get
    %profit
    g=transcoef*vars';