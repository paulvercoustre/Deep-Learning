[Dtrain,Dtest]  = load_digit7;

whos
[nsamples,ndimensions] = size(Dtrain);

meanDigit =0;
for d=1:nsamples
    meanDigit = meanDigit + Dtrain(d,:)/nsamples;
end
%% simpler & faster
meanDigit = mean(Dtrain,1)';
meanImage = reshape(meanDigit,[28,28]);
figure,imshow(meanImage);

covDigits = 0;
for d=1:nsamples
    covDigits = covDigits + ((Dtrain(d,:)-meanDigit')'*(Dtrain(d,:)-meanDigit'))/(nsamples-1);
end

covD = (Dtrain - meanDigit')*(Dtrain -meanDigit')'./nsamples;
figure,imagesc(covD);

covDigitsMatlab = cov(Dtrain);
%figure,imagesc(covDigits)
%% make sure covDigitsMatlab = your covDigits
figure,imagesc(covDigitsMatlab)

%% get top-5 eigenvectors
[eigvec,eigvals] = eigs(covDigitsMatlab,10);

%figure,
%subplot(1,3,1); imshow(reshape(eigvec(:,1),[28,28]),[])
%subplot(1,3,2); imshow(reshape(eigvec(:,2),[28,28]),[])
%subplot(1,3,3); imshow(reshape(eigvec(:,3),[28,28]),[])

for basis_idx = [1:2]
    factors =[-2,0,2];
    figure,
    for k=1:3
        imshow(reshape(meanDigit + 2*factors(k)*eigvec(:,basis_idx),[28,28]))
    end
end

%% calculating expansion coefficients / coordinates in eigenbasis
[tsamples,tdimensions] = size(Dtest);
[eigsample,eigdimensions] = size(eigvec);

expCoef = zeros(tsamples,eigdimensions);
for n=1:tsamples
    for d=1:eigdimensions
        expCoef(n,d) = eigvec(:,d)'*(Dtest(n,:)-meanDigit')';
    end
end

%% assessing the quality of our model with D from 1 to 10
error = zeros(1,eigdimensions);
for d=1:eigdimensions    
    for n=1:tsamples
        imageApprox=meanDigit;
        for k=1:d
            imageApprox = imageApprox + expCoef(n,k)*eigvec(:,k);           
        end 
        error(d) = error(d) + sqrt(dot((Dtest(n,:)'-imageApprox),(Dtest(n,:)'-imageApprox)));
    end 
end
   
plot(error);
xlabel('# of eigenvectors(D) ','fontsize',20); ylabel('E(D)','fontsize',20);
print('-djpeg','pca');