%%
%fix the labels
%used for classification task guided deep models
clear;clc;
%{
A=[%4 0 1 2 0 1 4 1 4 1 3 2 2 4 2 0 3 2 3 3 0 2 2 3 3 2 4 3 4 0 2 2 4 2 4 1 ...
    %4 1 3 3 2 0 4 1 1 2 0 1 3 1 3 4 4 2 0 4 2 3 4 1 3 2 3 2 2 2 1 1 3 2 0 3 ...
    %2 4 2 4 4 0 3 1 4 1 2 1 2 0 4 2 4 0 0 3 0 2 3 0 3 4 0 0 ...
    %0 2 2 2 2 2 4 3 2 2 1 2 2 0 3 3 2 4 2 4 0 2 4 2 3 0 3 3 2 3 3 2 0 2 3 2 3 ...
 %0 2 0 4 0 4 1 4 2 1 1 1 2 0 0 4 3 3 1 1 1 2 2 3 0 0 0 2 2 2 3 2 3 1 0 3 3 ...
 %0 2 3 3 0 1 3 3 4 3 1 0 0 1 4 4 3 2 2 4 4 3 2 3 2 1 ];
 2 3 2 3 3 3 0 3 2 2 4 3 4 3 1 1 3 2 4 4 3 3 3 3 2 3 2 4 4 2 3 2 1 0 1 3 0 ...
 1 1 3 1 0 4 1 4 4 4 2 2 0 3 3 4 2 0 1 2 2 2 3 4 4 1 2 3 0 3 1 2 1 4 4 0 0 ...
 4 4 1 2 3 4 2 4 4 3 2 2 4 2 2 1 4 4 1 2 2 4 2 2 3 4];

B=[%4 0 4 2 0 1 4 2 4 1 3 4 4 0 2 0 3 3 3 3 0 3 2 3 3 1 4 3 1 0 2 2 4 2 0 ...
    %2 2 0 3 3 2 1 1 4 1 4 0 1 3 1 2 1 4 2 4 4 3 2 1 0 2 2 3 1 0 2 1 1 3 3 ...
    %0 3 2 0 1 2 1 0 4 1 4 1 2 4 2 4 4 3 1 0 4 3 1 2 3 4 3 1 0 0 ...
    %1 1 4 1 2 4 0 3 1 1 1 1 2 0 3 2 2 0 2 4 0 2 4 0 3 1 3 1 1 3 3 1 0 1 3 1 3 ...
 %0 4 4 4 4 2 4 2 1 0 0 1 2 1 0 0 3 3 0 0 4 2 3 0 0 0 0 2 2 4 3 2 3 0 0 3 4 ...
 %4 2 3 3 0 1 2 3 1 4 2 0 0 4 1 1 3 2 2 4 4 3 2 2 2 1 ];
 0 2 3 2 1 3 0 3 4 3 0 2 0 3 0 2 3 2 4 2 3 2 3 3 2 2 2 1 0 1 1 3 0 1 1 3 0 ...
 4 2 3 1 1 4 1 4 1 1 2 2 4 1 3 4 4 4 0 2 4 1 3 4 4 0 1 3 1 3 0 4 1 0 1 4 1 ...
 3 4 4 4 3 1 2 0 4 3 4 2 4 3 2 4 4 4 2 1 3 1 4 2 3 0];
%}
A=[3 0 1 1 0 2 3 1 0 4 4 4 4 4 1 0 0 1 3 3 1 0 1 0 1 1 0 4 1 1 0 3 1 3 3 0 1 ...
 3 4 4 4 0 0 1 3 1 0 3 4 0 2 2 4 2 1 1 4 3 0 2 2 0 0 0 0 1 1 3 3 1 3 1 3 3 ...
 3 0 2 4 4 2 1 2 0 4 3 1 2 4 0 2 1 1 1 2 4 2 1 4 3 3];
B=[3 4 1 0 4 4 3 3 0 0 0 0 4 2 4 1 1 0 3 3 1 0 4 0 2 1 0 4 4 0 1 4 0 2 2 0 4 ...
 4 1 2 1 1 0 1 2 0 1 3 4 0 2 1 1 1 0 0 1 2 1 2 2 0 1 0 0 4 1 3 3 2 3 2 1 3 ...
 2 1 1 1 0 0 1 3 1 0 1 4 0 4 0 4 2 1 2 2 2 1 1 4 3 0];

for index = 1:100
    if(A(1,index)==4)
        A(1,index) = 0;
    else
        A(1,index) = A(1,index) + 1;
    end
end

for index = 1:100
    if(B(1,index)==4)
        B(1,index) = 0;
    else
        B(1,index) = B(1,index) + 1;
    end
end

mos = B;
iqa = A;

%%
%logistic fitting

addpath('/home/chezhaohui/QA/Fitting/xiongkuo/');
[delta,beta,yhat,y,diff] = findrmse2(iqa,mos);
iqa = yhat';
beta
rmpath('/home/chezhaohui/QA/Fitting/xiongkuo/');

%%
%compute SRCC, PLCC and RMSE
addpath('/home/chezhaohui/QA/Fitting/');
[plcc,srocc,rmse] = verify_performance(mos, iqa, beta);
%beta = [10, 0, mean(iqa), 0.1, 0.1];[plcc,srocc,rmse] = verify_performance(mos, iqa, beta);
rmpath('/home/chezhaohui/QA/Fitting/');
