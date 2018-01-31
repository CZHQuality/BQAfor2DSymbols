%this script is used to calculate the pearson linear correlation
%coefficient and root mean sqaured error after regression

%get the objective scores computed by the IQA metric and the subjective
%scores provided by the dataset

%function [plcc,srocc,rmse] = verify_performance(mos,predict_mos)
function [plcc,srocc,rmse] = verify_performance(varargin)

mos = varargin{1};
predict_mos = varargin{2};

predict_mos = predict_mos(:);
mos = mos(:);

if (isempty(varargin{3}))
%initialize the parameters used by the nonlinear fitting function
    beta(1) = 10;
    beta(2) = 0;
    beta(3) = mean(predict_mos);
    beta(4) = 0.1;
    beta(5) = 0.1;
else
    beta(1) = varargin{3}(1);
    beta(2) = varargin{3}(2);
    beta(3) = varargin{3}(3);
    beta(4) = varargin{3}(4);
    beta(5) = varargin{3}(5);
end



beta

%fitting a curve using the data
[bayta ehat,J] = nlinfit(predict_mos,mos,@logistic,beta);
%given a ssim value, predict the correspoing mos (ypre) using the fitted curve
[ypre junk] = nlpredci(@logistic,predict_mos,bayta,ehat,J);
% ypre = predict(logistic,fsimValues,bayta,ehat,J);
rmse = sqrt(sum((ypre - mos).^2) / length(mos));%root meas squared error
plcc = corr(mos, ypre, 'type','Pearson'); %pearson linear coefficient
srocc = corr(mos, predict_mos, 'type','spearman');

end
