%% load data
[Dtrain,Dtest]  = load_digit7;
[trainsamples,trainndimensions] = size(Dtrain);
[testsamples,testndimensions] = size(Dtest);

%% parameters
% for kmeans
K = [2,3,4,5,10,50,100]; % performing kmeans with required values of k
      
max_numIteration = 300; % max number of iterations to stop algo just in case
training_data = Dtrain; % choose the data on which to run the algo. 
testing_data = Dtest; % choose the data on which to test distortion cost.

% for computation
number_eval = 10; % number of times we compute kmeans to avoid local minima issue

% we add a new colum corresponding to the centroid the point is assigned to. initialized at 0
w_labels = zeros(size(training_data,1),1); % Set at O
data = [training_data, w_labels];

w_labels_test = zeros(size(testing_data,1),1);
test = [Dtest,w_labels_test];

% we create empty arrays to store distortion cost
store_train_distortion = [];
store_test_distortion = [];

%% kmeans
for k = K % number of centroids
    store_final_Centroid = zeros(k,size(data,2)-1,number_eval); % creating multidim array to store finalCentroids of all kmeans iterations
    store_distortion = zeros(number_eval,max_numIteration); % creating a matrix to store distortion of all kmeans iterations

    for eval = 1:number_eval % setting up a loop of kmeans instances
        [final_Centroid,distortion,total_iterations] = kmeans_function(k,data,max_numIteration);
        store_final_Centroid(:,:,eval) = final_Centroid; % storing the centroid coordinates for each kmeans instance
        store_distortion(eval,:) = distortion; % storing the distortion values for each kmeans instance
    end

    % finding best converging distortion instance
    non_zero_store_distortion = store_distortion; 
    non_zero_store_distortion(non_zero_store_distortion == 0) = 1e10; % replacing zero values with large values to perform min 
    [~,best_idx] = find(min(non_zero_store_distortion(:))); % finding the coordinates of the minimal distortion instance
    best_distortion = non_zero_store_distortion(best_idx,:);
    best_distortion(best_distortion==1e10) = [];

    % plot distortion
    figure,
    plot(best_distortion);
    xlabel('Iterations');
    ylabel('Distortion');
    title(['Smallest Distortion Instance - k = ',num2str(k),'',]);
    %print('-djpeg','kmeans_distortion_k3');

    % find the centroids associated with that instance and plot them
    best_centroid = store_final_Centroid(:,:,best_idx);

    % show resulting digit clusters
    figure,
    for centroid_plot = 1:k
        subplot(1,k,centroid_plot); imshow(reshape(best_centroid(centroid_plot,:),[28,28]),[]);
        %title(sprintf('centroid %i', centroid_plot));
    end
    %print('-djpeg','kmeans_clusters_k3');
    
    
    %% perform kmeans on testing data with training set centroids
    test_dist = kmeans_distance_calculation(best_centroid, test); % calculate distance
    test(:,size(test,2)) = kmeans_find_minimal_distance(test_dist); % find centroid with min distance
    test_centroids = kmeans_update_centroids(test,test(:,size(test,2)),best_centroid); % assign centroid with min distance
    % calculate distortion cost
    test_distortion = zeros(1,1);
    for i=1:size(test,1)
        test_distortion = test_distortion + norm(test(i,1:size(test,2)-1) - test_centroids(test(i,size(test,2)),:))^2;
    end
    store_test_distortion(k) = test_distortion;
    store_train_distortion(k) = min(best_distortion);
end

store_test_distortion(store_test_distortion == 0) = [];
store_train_distortion(store_train_distortion == 0) = [];

figure, 
plot(K,store_test_distortion,'-^');
hold on;
plot(K,store_train_distortion,'--o');
hold off
xlabel('Number of Centroids');
ylabel('Distortion Cost');
legend('Testing Data','Training Data');