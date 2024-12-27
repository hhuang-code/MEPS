clc, clear all, close all;
addpath(genpath('fastAuction_v2.5'));

X = load('result/faust/prob.mat');
prob_list = double(X.prob);
pred = nan(size(prob_list, 1), size(prob_list, 2));
tic;
for i=1:size(prob_list, 1)
    prob = prob_list(i, 1:end, 1:end);
    prob = reshape(prob, 6890, 6890);
    assignment = assignmentAlgs(prob, 'auction');
    xin = 1:size(prob, 1);
    yin = assignment(xin);            
    pred(i, :) = yin;
    disp(['done', num2str(i)]);
end
toc;
save('result/faust/result_LAP.mat', 'pred')