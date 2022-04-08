format long

a = 2;   % initial interval for root
b = 4;   % initial interval for root
N = 60;  % number of iterations
i = 0;
lamb  = zeros(1,N); % estimates of lambda
alpha = zeros(1,N); % estimates of order alpha

% initalize |pn-p|-type differences
diffA = 2; diffB = 2; diffC = 2;
p = 2;

while i<N
    pNew  = a+(b-a)/2;
    diffC = diffB;
    diffB = diffA;
    diffA = abs(pNew-p);
    p = pNew; 
    if f1(a)*f1(p) > 0
        a=p;
    else
        b=p;
    end
    i = i+1;
    alpha(i) = log(diffA/diffB)/log(diffB/diffC);
    lamb(i)  = diffA/(diffB^alpha(i));
    display([i,p,lamb(i),alpha(i)]);
end

figure(1)
plot(1:N,alpha,'*-','LineWidth',2)
xlabel('Number of iterations')
ylabel('Estimate of \alpha')
set(gca,'FontSize',12)