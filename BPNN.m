clear;
Train = load('Data_train.txt');
Test =load('Data_test.txt');
 [train_row, train_col] = size(Train);
%%���ýڵ���[�����,xx,...,xx,�����],��������
Node = [train_col-1 ,100,100,1];
layers = size(Node,2);
step = [0.001, 0.001,0.001];
lambda = [0.135 0.118 0.15];
alpha=0.15;
iter = 5000;
G_Node=gpuArray(single(Node));
G_step =gpuArray(single(step));
G_lambda=gpuArray(single(lambda));
G_iter=gpuArray(single(iter));
G_alpha=gpuArray(single(alpha));
%%----------��������----------------------------
Train = single(Train);
Test = single(Test);
train_ans = Train( : , train_col);
test_ans=Test( : , train_col);
test = mapminmax(Test( : , 1 : train_col - 1)',0,1)';
train  = mapminmax(Train( : , 1 : train_col - 1)',0,1)';%��һ��
%%------��֤
% G_train=gpuArray(train(1:7922,:));
% G_train_ans=gpuArray(train_ans(1:7922,:));
% G_val=gpuArray(train(7922:train_row,:));
% G_val_ans=gpuArray(train_ans(7922:train_row,:));
%%-----����
G_train=gpuArray(train);
G_train_ans=gpuArray(train_ans);
G_val=gpuArray(test);
G_val_ans=gpuArray(test_ans);
[MSE,corr,MSE_val,corr_val,out1,out2]= BPNN(G_train, G_val, G_iter,G_Node, layers,G_step,G_lambda,G_alpha,G_train_ans,G_val_ans,size(G_train_ans,1),size(G_val_ans,1));
figure;
plot(1:iter, MSE, 'b');hold on
plot(1:iter, MSE_val, 'r');hold off
title('MSE');xlabel('��������'),ylabel('MSE');legend('ѵ����','��֤��');grid on;
figure;
plot(1:iter, corr, 'b');hold on
plot(1:iter, corr_val, 'r');hold off
title('�����');xlabel('��������'),ylabel('�����');legend('ѵ����','��֤��');grid on;
figure;
stem(1:size(G_train_ans,1), G_train_ans, 'b','Marker','none');hold on
stem(1:size(G_train_ans,1), out1, 'r','Marker','none');hold off
title('����Ա�');xlabel('1-11��'),ylabel('cnt');legend('ʵ��ֵ','ģ�ͽ��');grid on;
figure;
stem(1:size(G_val_ans,1), G_val_ans, 'b','Marker','none');hold on
stem(1:size(G_val_ans,1), out2, 'r','Marker','none');hold off
title('����Ա�');xlabel('12��'),ylabel('cnt');legend('ʵ��ֵ','ģ�ͽ��');grid on;
xlswrite('final_ans.xlsx',round(gather(out2)));%���
%%-------K�۴���-------------------------------
% K_Fold(10, Train, G_iter, G_Node, layers,G_step,G_lambda);







function [] = K_Fold(K, Train, iter, Node, layers,step,lambda,alpha)
    train = gpuArray(single(Train));
    [train_row, train_col] = size(Train);
    G_train_row=gpuArray(single(train_row));
    k=gpuArray(single(K));
    Indices = gpuArray(single(crossvalind('Kfold', G_train_row, K)));
    train_ans = train( : , train_col);
    train  = mapminmax(train( : , 1 : train_col - 1)',0,1)';%��һ��
    MSE_K=gpuArray.zeros(1, k,'single');
    corr_K=gpuArray.zeros(1, k,'single');
    MSE_val_K=gpuArray.zeros(1, k,'single');
    corr_val_K=gpuArray.zeros(1, k,'single');
    for i=1:k %K�۽�����֤
        i
        val = train((Indices == i),: );
        tra = train((Indices ~= i),: );
        val_ans = train_ans((Indices == i),: );
        tra_ans = train_ans((Indices ~= i),: );
        [MSE,corr,MSE_val,corr_val,~,~]= BPNN(tra, val, iter,Node, layers,step,lambda,alpha,tra_ans,val_ans,size(tra_ans,1),size(val_ans,1));
        %��10��ƽ��
        MSE_K = MSE_K + MSE / k;
        corr_K = corr_K + corr / k;
        MSE_val_K = MSE_val_K + MSE_val / k;
        corr_val_K = corr_val_K + corr_val / k;
    end
    figure;
    plot(1:iter, MSE_K, 'b');hold on
    plot(1:iter, MSE_val_K, 'r');hold off
    title('MSE');xlabel('��������'),ylabel('F1');legend('ѵ����','��֤��');grid on;
    figure;
    plot(1:iter, corr_K, 'b');hold on
    plot(1:iter, corr_val_K, 'r');hold off
    title('�����');xlabel('��������'),ylabel('F1');legend('ѵ����','��֤��');grid on;
end
function [MSE,corr,MSE_val,corr_val, out1, out2] = BPNN(train, val, iter, Node, layers, step, lambda,alpha,train_ans, val_ans,train_row,val_row)
    w = cell(1, layers-1);
    out = cell(1, layers - 1);
    out_val = cell(1, layers - 1);
    delta = cell(1, layers - 1);
    theta = cell(1, layers - 1);
    layers=gpuArray(single(layers));
    MSE=gpuArray.zeros(1, iter,'single');
    MSE_val=gpuArray.zeros(1, iter,'single');
    corr=gpuArray.zeros(1,iter,'single');
    corr_val=gpuArray.zeros(1,iter,'single');
    %%--------------��ʼ��Ȩֵ����----------------
    for i = 1 :layers-1
        w{i} = gpuArray.rand(Node(i), Node(i+1),'single') - 0.5;
        theta{i} = gpuArray.ones(1,Node(i+1),'single') * 0.1;
    end
    %%--------------����----------------------------
    for it = 1 : iter
        %%------------���������--------------------
        %�����->���ز�
        out{1} = 1 ./ (1 + exp(-train *w{1} + theta{1}));%ѵ����
        out_val{1} = 1 ./ (1 + exp(-val *w{1} + theta{1}));%��֤��
        %���ز�
        for i=2 : layers-2
             out{i} =1 ./ (1 + exp(-out{i - 1} * w{i} + theta{i}));%ѵ����
             out_val{i} = 1 ./ (1 + exp(-out_val{i - 1} * w{i} + theta{i}));%��֤��
        end
        %�����
        out{layers-1} =  out{layers-2} * w{layers-1}  + theta{layers-1};%ѵ����
        out{layers-1}(out{layers-1}<0)=out{layers-1}(out{layers-1}<0)*alpha;%rule
        out_val{layers-1} = out_val{layers-2} * w{layers-1} + theta{layers-1};%��֤��
        out_val{layers-1}(out_val{layers-1}<0)=out_val{layers-1}(out_val{layers-1}<0)*alpha;%rule
        %%-----------���򴫲��㷨,�������delta-----
        %�����delta
        Err = train_ans - out{layers-1};
        Err(Err<0)=Err(Err<0)*alpha;
        Err_val =  val_ans - out_val{layers-1};
        Err_val(Err_val<0)=Err_val(Err_val<0)*alpha;
        delta{layers-1} = (Err);
        %���ز�delta
        for i=layers-2 : -1 : 1
            delta{i} = out{i} .* (1-out{i}) .* (delta{i+1} * w{i+1}');
        end
        %%-----------------------����w----------------
        %���ز�
        for i = layers-1 : -1 : 2
            w{i} = w{i} + step(i) *  (out{i-1}'*delta{i} / train_row  + lambda(i)  * w{i});
            theta{i} = theta{i} + step(i)  / train_row * sum(delta{i},1) ;
        end
        %�����->���ز�
        w{1} = w{1} + step(1)  *  (train' *delta{1} / train_row+ lambda(1)  * w{1});
        theta{1} = theta{1} + step(1)  / train_row * sum(delta{1},1) ;
        %%-----------------�������-------------------
        MSE(it) = Err' * Err / train_row;
        MSE_val(it) =  Err_val' * Err_val / val_row;
        tmp=corrcoef(out{layers-1},train_ans);
        corr(it) = tmp(2,1);
        tmp=corrcoef(out_val{layers-1},val_ans);
        corr_val(it) = tmp(2,1);
    end
    out1 = out{layers-1};
    out2 = out_val{layers-1} ;
end
