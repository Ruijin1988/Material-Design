% load 2nd layer hidden state
load('WB_nowh_hidstate_1layer(2f6ws12rP20Pb01).mat');
% 1st layer hidenstate pooling
temp=[];temp2=[];temp3=[];
%pooling ratio
C=2;
%% pooling process
for i = 1:100
    for j = 1:24
        temp = xtr(i,:);
        temp = reshape(temp,[195*195,24]);
        temp = reshape(temp(:,j),[sqrt(size(temp,1)),sqrt(size(temp,1))]);
        temp = imresize(temp,[97 97]);
        temp = double(im2bw(temp,0.2));
        
        temp2(:,j) = temp(:);
        
        
    end
    hidstate=temp2;
        fname = sprintf('hidstate_1stlayer(poolratio02)_(2f40f6ws9ws12rP20P10Pb01)_%d',i);
        save(sprintf('%s.mat',fname),'hidstate', '-v7.3');
    
    temp3(i,:) = temp2(:);
end
pool = temp3;

%% pooling back for comparison
temp=[];temp2=[];temp3=[];
for i = 1:100
    for j = 1:24
        temp = pool(i,:);
        temp = reshape(temp,[97*97,24]);
        temp = reshape(temp(:,j),[sqrt(size(temp,1)),sqrt(size(temp,1))]);
        temp = imresize(temp,[195 195]);
        temp = double(im2bw(temp));
        
        temp2(:,j) = temp(:);
        
        
    end
    hidstate=temp2;
%         fname = sprintf('TEST_%d',i);
%         save(sprintf('%s.mat',fname),'hidstate', '-v7.3');
    
    temp3(i,:) = temp2(:);
end
pool_back1 = temp3;