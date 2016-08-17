% clear;
% addpath('function_code','utils');
% 
% load('filter_4th_layer_288.mat');
% load('rbm_1to100_4thlayer_(2f40f288f1000f6ws9ws9ws36ws12rP20P10P10P10Pb01)_alloy_w36_b1000_trans_ntx1_gr1_pb0.1_pl10_iter_2400.mat')
numch = 1000;
% 
%define 3rd layer filter
W_Five=gather(weight.vishid);
% pool back 3rd layer filter
% for i = 1:size(W_Five,2)   
%     W_temp=reshape(W_Five(:,i),[36*36 288]);
%     for j = 1:numch
%         W_temp2=reshape(W_temp(:,j),[36 36]);
%         W_temp2=imresize(W_temp2,[36*4 36*4]);
%         W_temp3(:,j)=W_temp2(:);
%     end
%     W_Five_Pool(:,i)=W_temp3(:);
% end
% clear W_temp3;clear W_temp2;clear W_temp;
% W_F=W_Five_Pool;


% define 2nd layer filter
filter_t_corr=store_filter4;

% filter_t_corr=reshape(filter_t_corr,[numel(filter_t_corr)/numch,numch]);
for j = 1:size(W_Five,2)
    filter_4th_temp=W_Five(:,j);

negdata = zeros(36*4, 36*4);
for i=1:numch
%     for ii =1:size(filter_t,2)
        filter_t_temp=filter_t_corr(:,i);
        filter_t_temp=reshape(filter_t_temp,[sqrt(size(filter_t_corr,1)),sqrt(size(filter_t_corr,1))]);
        filter_4th = filter_4th_temp((i-1)*size(W_Five,1)/numch+1:i*size(W_Five,1)/numch,:);
        filter_4th=reshape(filter_4th,[sqrt(size(W_Five,1)/numch),sqrt(size(W_Five,1)/numch)]);
%         filter_t_temp=abs(filter_t_temp-1); %flip the filter, display 0.25~1
%         temp2=conv2(filter_t_temp,filter_4th,'same');
        temp3=conv2(filter_4th,filter_t_temp,'same');
%         figure(18),subplot(12,12,i),imshow(temp3,[-5 5])
        negdata = temp3+negdata;
%         negdata=sigmoid(negdata);
%         figure(5+10);subplot(10,10,i),imshow(negdata);
%         figure(j+10); subplot(1,8)imshow(negdata) negdata = temp2+negdata;
%     end
end
store_filter5(:,j)=negdata(:);
% figure(104); subplot(12,12,j),display_network(reshape(negdata,size(negdata,1)*size(negdata,2),1));

end
