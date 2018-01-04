images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

image = images(:,1);
label = labels(1);

A = zeros(28,28);
counter=1;
for j=1:28
    for i=1:28
        A(i,j)=image(counter);
        counter = counter+1;
    end
end
imshow(A);

sum(sum(images(:,1:1000) > 0))