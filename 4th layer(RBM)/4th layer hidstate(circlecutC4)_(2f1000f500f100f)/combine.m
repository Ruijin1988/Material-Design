for ii = 1:80
fname = sprintf('hidstates4th_circlecutC4_(2f1000f500f100f12ws47ws1ws1ws)_%d',ii);
f1=load(sprintf('%s.mat', fname));   
temp = double([f1.hidstate;]);
temp = permute(temp,[3,2,1]);
xtr(ii,:) = temp(:)';
% f1=load([CIFAR_DIR '/filter8_ws12.mat']);
end
fname = sprintf('hidstates4th_hardsphere_(p2p2)_imresize_(24f40f288f1000f6ws9ws9ws36ws)');
save(sprintf('%s.mat',fname),'xtr', '-v7.3');