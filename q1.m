% This program generates data set D 

clear all;
close all;
clc;

%----------Generating Training data----------------------------------
mu = 0;
sigma = 0.1;
L = 200;

E = sigma*randn(L,1)+mu;

M = 200;    % number of rows
N = 2;      % number of columns

D = zeros(M,N); % data set matrix

for i = 1:M
    for j = 1:N
        if j == 1
            D(i,1) = (i-1)/(M-1);
        else
            D(i,2) = sin(2*pi*D(i,1)) + E(i);
        end
    end
end
%----------------------Generating sine with data------------------------
sine = sin(2 * pi * D(:,1));
plot(D(:,1),sine,'-','MarkerSize',10,'color','g');
hold on;
plot(D(:,1),D(:,2),'o','MarkerSize',5,'color','b');          
title('Training examples');
xlabel('x coordinate');
ylabel('y coordinate');
hold off;
%-----------------------------------------------------------------------

shuffle = D(randperm(size(D,1)),:); % Shuffling data for random selection

%output_mse = zeros(125,4);
data_size = [10 50 100 150 200];
lambda = [0 0.01 0.02 0.2 0.26];
order = [0 1 3 4 9];
x = 1; %output incrementer

for i = 1:5 %loop for lambda
    
    for j = 1:5 %loop for size
        

        for k = 1:5 %loop for order
            
            X = shuffle(1:data_size(j), 1); %---------------Training set X size
            Y = shuffle(1:data_size(j), 2); %---------------Training set Y size
            %sine = sin(2 * pi * X);
            m = length(Y); % training set size
            %----------------------------------------------------------------------



            %--------------------testing data prep---------------------------------
            Xtrain = shuffle(1:199,1);
            Xtest = Xtrain + (1/(2*(M-1)));
            Ytest = sin(2*pi*Xtest);
            mtest = length(Ytest);
            %----------------------------------------------------------------------



            %----------------Setting Model Parameters------------------------------

            ORDER = order(k);
            l = lambda(i);

            theta = zeros(ORDER+1,1); %-------------------initial weight
            
            %X = [ones(m,1) X X.^2 X.^3];%--------Polynomial feature sizing
            X = poly(X,ORDER);
            %Xtest = [ones(mtest,1) Xtest Xtest.^2 Xtest.^3];
            Xtest = poly(Xtest,ORDER);
            %---------------------------------------------------------------------



            %-------------------Running Normal Equation---------------------------
            theta = normalEqn(X,Y,l);
            %hold on;
            %plot(X(:,2), X * theta, 'o','Color','red');
            %plot(Xtest(:,2), Xtest * theta, 'o','Color','red');
            %legend('testing data', 'Linear regression')
            %hold off;
            %---------------------------------------------------------------------



            %-----------------------Computing MSE---------------------------------
            MSE_train = computeCost(X, Y, theta);
            MSE_test = computeCost(Xtest, Ytest, theta);
            output(x,:) = [l ORDER data_size(j) MSE_train];
            output_test(x,:) = [l ORDER data_size(j) MSE_test];
            x = x + 1;
            
        end
    end
end
            %---------------------------------------------------------------------

l1_table = sortrows(output(1:25,:),2);
l2_table = sortrows(output(26:50,:),2);
l3_table = sortrows(output(51:75,:),2);
l4_table = sortrows(output(76:100,:),2);
l5_table = sortrows(output(101:125,:),2);

l1_table_test = sortrows(output_test(1:25,:),2);
l2_table_test = sortrows(output_test(26:50,:),2);
l3_table_test = sortrows(output_test(51:75,:),2);
l4_table_test = sortrows(output_test(76:100,:),2);
l5_table_test = sortrows(output_test(101:125,:),2);

%----------------------Training Plot------------------------------------

figure(1);
hold on;

plot(l1_table(1:5,3),l1_table(1:5,4),'x-','Color','r');%order 0
plot(l1_table(6:10,3),l1_table(6:10,4),'x-','Color','g');%order 1
plot(l1_table(11:15,3),l1_table(11:15,4),'x-','Color','b');%order 3
plot(l1_table(16:20,3),l1_table(16:20,4),'x-','Color','y');%order 4
plot(l1_table(21:25,3),l1_table(21:25,4),'x-','Color','c');%order 9
title('MSE train plot Lambda = 0');
legend('Order0', 'Order1', 'Order3', 'Order4', 'Order9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(2);
hold on;

plot(l2_table(1:5,3),l2_table(1:5,4),'x-','Color','r');
plot(l2_table(6:10,3),l2_table(6:10,4),'x-','Color','g');
plot(l2_table(11:15,3),l2_table(11:15,4),'x-','Color','b');
plot(l2_table(16:20,3),l2_table(16:20,4),'x-','Color','y');
plot(l2_table(21:25,3),l2_table(21:25,4),'x-','Color','c');
title('MSE train plot Lambda = 0.01');
legend('Order0', 'Order1', 'Order3', 'Order4', 'Order9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(3);
hold on;

plot(l3_table(1:5,3),l3_table(1:5,4),'x-','Color','r');
plot(l3_table(6:10,3),l3_table(6:10,4),'x-','Color','g');
plot(l3_table(11:15,3),l3_table(11:15,4),'x-','Color','b');
plot(l3_table(16:20,3),l3_table(16:20,4),'x-','Color','y');
plot(l3_table(21:25,3),l3_table(21:25,4),'x-','Color','c');
title('MSE train plot Lambda = 0.02');
legend('Order0', 'Order1', 'Order3', 'Order4', 'Order9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(4);
hold on;

plot(l4_table(1:5,3),l4_table(1:5,4),'x-','Color','r');
plot(l4_table(6:10,3),l4_table(6:10,4),'x-','Color','g');
plot(l4_table(11:15,3),l4_table(11:15,4),'x-','Color','b');
plot(l4_table(16:20,3),l4_table(16:20,4),'x-','Color','y');
plot(l4_table(21:25,3),l4_table(21:25,4),'x-','Color','c');
title('MSE train plot Lambda = 0.2');
legend('Order0', 'Order1', 'Order3', 'Order4', 'Order9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(5);
hold on;

plot(l5_table(1:5,3),l5_table(1:5,4),'x-','Color','r');
plot(l5_table(6:10,3),l5_table(6:10,4),'x-','Color','g');
plot(l5_table(11:15,3),l5_table(11:15,4),'x-','Color','b');
plot(l5_table(16:20,3),l5_table(16:20,4),'x-','Color','y');
plot(l5_table(21:25,3),l5_table(21:25,4),'x-','Color','c');
title('MSE train plot Lambda = 0.26');
legend('Order0', 'Order1', 'Order3', 'Order4', 'Order9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

%--------------------------Testing Plot------------------------------------

figure(6);
hold on;

plot(l1_table_test(11:15,3),l1_table_test(11:15,4),'x-','Color','b');%order 3
plot(l1_table(11:15,3),l1_table(11:15,4),'o-','Color','r');%order 3
plot(l1_table_test(21:25,3),l1_table_test(21:25,4),'x-','Color','g');%order 9
plot(l1_table(21:25,3),l1_table(21:25,4),'o-','Color','c');%order 9
title('MSE test vs train plot Lambda = 0');
legend('TestOrder3', 'TrainOrder3', 'TestOrder9', 'TrainOrder9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(7);
hold on;

plot(l2_table_test(11:15,3),l2_table_test(11:15,4),'x-','Color','b');
plot(l2_table(11:15,3),l2_table(11:15,4),'o-','Color','r');
plot(l2_table_test(21:25,3),l2_table_test(21:25,4),'x-','Color','g');
plot(l2_table(21:25,3),l2_table(21:25,4),'o-','Color','c');
title('MSE test vs train plot Lambda = 0.01');
legend('TestOrder3', 'TrainOrder3', 'TestOrder9', 'TrainOrder9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(8);
hold on;

plot(l3_table_test(11:15,3),l3_table_test(11:15,4),'x-','Color','b');
plot(l3_table(11:15,3),l3_table(11:15,4),'o-','Color','r');
plot(l3_table_test(21:25,3),l3_table_test(21:25,4),'x-','Color','g');
plot(l3_table(21:25,3),l3_table(21:25,4),'o-','Color','c');
title('MSE test vs train plot Lambda = 0.02');
legend('TestOrder3', 'TrainOrder3', 'TestOrder9', 'TrainOrder9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(9);
hold on;

plot(l4_table_test(11:15,3),l4_table_test(11:15,4),'x-','Color','b');
plot(l4_table(11:15,3),l4_table(11:15,4),'o-','Color','r');
plot(l4_table_test(21:25,3),l4_table_test(21:25,4),'x-','Color','g');
plot(l4_table(21:25,3),l4_table(21:25,4),'o-','Color','c');
title('MSE test vs train plot Lambda = 0.2');
legend('TestOrder3', 'TrainOrder3', 'TestOrder9', 'TrainOrder9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;

figure(10);
hold on;

plot(l5_table_test(11:15,3),l5_table_test(11:15,4),'x-','Color','b');
plot(l5_table(11:15,3),l5_table(11:15,4),'o-','Color','r');
plot(l5_table_test(21:25,3),l5_table_test(21:25,4),'x-','Color','g');
plot(l5_table(21:25,3),l5_table(21:25,4),'o-','Color','c');
title('MSE test vs train plot Lambda = 0.26');
legend('TestOrder3', 'TrainOrder3', 'TestOrder9', 'TrainOrder9')
xlabel('Dataset size');
ylabel('MSE');
xticks([0 10 50 100 150 200]);

hold off;
        


