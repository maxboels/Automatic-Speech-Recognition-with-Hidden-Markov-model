close all
delete all
%Emitting states
N = 2;
%Number of observations
T = 7;
%State Transition Probabilities, {pi, aij, ni}.
A = [0 0.93 0.07 0;
     0 0.74 0.21 0.05;
     0 0.08 0.90 0.02;
     0 0    0    0];
%Output Probabilities densities, bi(ot) (Mean and Variance).
P = [3   5; 
    1.21 0.25]; 
states = 2;
mu = P(1,:);
sigma = P(2, :);
%Observations sequence, Ot
O = [1.8 2.6 2.7 3.3 4.4 5.4 5.2];

% Train the HMM with Expectation-Maximization with Baum-Welch equations.
%1. Draw the State Topology and output Prob. Density Functions (pdf)
figure(1)
X = 0:0.1:7;
Y = normpdf(X, mu(1), sigma(1));
plot(X, Y, 'b')
hold on
Y = normpdf(X, mu(2), sigma(2));
plot(X, Y, 'r')
hold on
scatter(1:7,biOt(1,:), 'b');
hold on
scatter(1:7, biOt(2,:), 'r')
xlabel("Observations");
ylabel("Prob(Ot)")
title("probability density functions")
legend("State 1", "State 2", "b1Ot", "b2Ot")
%2. Output density probability for each time frame and state
for i=1:states(1)
    for t = 1:T
    biOt(i,t) = (1/sqrt(2*pi*sigma(i)))*exp((-(O(t)-mu(i))^2)/(2*sigma(i)));
    end
end

%3. Forwards likelihoods and
%overall likelihood of the observations, for all T from t=1 to t=7
%Initialize at t=1
alpha(1,1)= A(1,2)*biOt(1,1); %Entry Probability to state 1
alpha(2,1)= A(1,3)*biOt(2,1); %Entry Probability to state 2
%Recur t ={2:7}
alpha(1,2)= (alpha(1,1)*A(2,2)+ alpha(2,1)*A(3,2))*biOt(1,2);
alpha(2,2)= (alpha(2,1)*A(3,3)+ alpha(1,1)*A(2,3))*biOt(2,2);

alpha(1,3)= (alpha(1,2)*A(2,2)+ alpha(2,2)*A(3,2))*biOt(1,3);
alpha(2,3)= (alpha(2,2)*A(3,3)+ alpha(1,2)*A(2,3))*biOt(2,3);

alpha(1,4)= (alpha(1,3)*A(2,2)+ alpha(2,3)*A(3,2))*biOt(1,4);
alpha(2,4)= (alpha(2,3)*A(3,3)+ alpha(1,3)*A(2,3))*biOt(2,4);

alpha(1,5)= (alpha(1,4)*A(2,2)+ alpha(2,4)*A(3,2))*biOt(1,5);
alpha(2,5)= (alpha(2,4)*A(3,3)+ alpha(1,4)*A(2,3))*biOt(2,5);

alpha(1,6)= (alpha(1,5)*A(2,2)+ alpha(2,5)*A(3,2))*biOt(1,6);
alpha(2,6)= (alpha(2,5)*A(3,3)+ alpha(1,5)*A(2,3))*biOt(2,6);

alpha(1,7)= (alpha(1,6)*A(2,2)+ alpha(2,6)*A(3,2))*biOt(1,7);
alpha(2,7)= (alpha(2,6)*A(3,3)+ alpha(1,6)*A(2,3))*biOt(2,7);
%Finalise
Pfoverall = (alpha(1,7)*A(2,4)+ alpha(2,7)*A(3,4));

%4. Backward likelihoods
%Initialize at t=T
beta(1,7)= A(2,4);
beta(2,7)= A(3,4);

beta(1,6)= beta(1,7)*A(2,2)*biOt(1,7)+ beta(2,7)*A(2,3)*biOt(2,7);
beta(2,6)= beta(2,7)*A(3,3)*biOt(2,7)+ beta(1,7)*A(3,2)*biOt(1,7);

beta(1,5)= beta(1,6)*A(2,2)*biOt(1,6)+ beta(2,6)*A(2,3)*biOt(2,6);
beta(2,5)= beta(2,6)*A(3,3)*biOt(2,6)+ beta(1,6)*A(3,2)*biOt(1,6);

beta(1,4)= beta(1,5)*A(2,2)*biOt(1,5)+ beta(2,5)*A(2,3)*biOt(2,5);
beta(2,4)= beta(2,5)*A(3,3)*biOt(2,5)+ beta(1,5)*A(3,2)*biOt(1,5);

beta(1,3)= beta(1,4)*A(2,2)*biOt(1,4)+ beta(2,4)*A(2,3)*biOt(2,4);
beta(2,3)= beta(2,4)*A(3,3)*biOt(2,4)+ beta(1,4)*A(3,2)*biOt(1,4);

beta(1,2)= beta(1,3)*A(2,2)*biOt(1,3)+ beta(2,3)*A(2,3)*biOt(2,3);
beta(2,2)= beta(2,3)*A(3,3)*biOt(2,3)+ beta(1,3)*A(3,2)*biOt(1,3);

beta(1,1)= beta(1,2)*A(2,2)*biOt(1,2)+ beta(2,2)*A(2,3)*biOt(2,2);
beta(2,1)= beta(2,2)*A(3,3)*biOt(2,2)+ beta(1,2)*A(3,2)*biOt(1,2);
% Finally,
Pbackward = (A(1,2)*beta(1,1)*biOt(1,1) +A(1,3)*beta(2,1)*biOt(2,1));

%5. Occupations likelihood Baum-Wemch re-estimation
for i = 1:states(1)
    for t = 1:T
Gamma(i,t)= (alpha(i,t)*beta(i,t))/Pfoverall;
    end
end

%6. Re-estimated Means and variance
%re-mean
for i = 1:states(1)
    for t = 1:T
    nmean(i,t)= Gamma(i,t)*O(t);
    dmean(i,t)= Gamma(i,t);
    end
    remean(i)=sum(nmean(i,:))/sum(dmean(i,:));
end
%re-variance
for i = 1:states(1)
    for t = 1:T
    nvariance(i,t) = (Gamma(i,t)*(O(t)-mu(i))*((O(t)-mu(i))'));
    end
    revariance(i)= sum(nvariance(i,:))/sum(Gamma(i,:));
end 

%7. Plots of the PDFs and Comments
X = 0:0.1:7;
Y = normpdf(X, remean(1), revariance(1));
plot(X, Y, 'b--')
hold on
Y = normpdf(X, remean(2), revariance(2));
plot(X, Y, 'r--')
hold on
% plot(1:7, B(2,:)) % add B to pdfs
xline(4.25, 'g--');
xlabel("Observations");
ylabel("Prob(Ot)")
title("Original and Trained probability density functions")
legend("Original State 1", "Original State 2", "Trained State 1", "Trained State 2", "Boundary")
hold off