%% performs kmeans function

function [ao_centroid,ao_disto,w_iter] = kmeans_function(ai_K,ai_Data,ai_maxIteration)

%% pick 'ai_k' random centroids
ao_centroid = kmeans_initialise_centroids(ai_K, ai_Data);   % selecting initial centroids

%% update centroid
w_Data = ai_Data; % making a copy of the data
w_Data(:,size(ai_Data,2)) = zeros(size(ai_Data,1), 1);	% setting labels to 0
ao_disto = zeros(1,ai_maxIteration); % creating a zero array to save iteration distortion
convergence = 0; % setting initial convergence to 0
w_iter = 1; % setting initial number of iterations to 1 

while convergence ~= 1   
    
    % calculating distance
    w_distance = kmeans_distance_calculation(ao_centroid, ai_Data);
    
    % finding the minimum value 
    w_Data(:,size(ai_Data,2)) =  kmeans_find_minimal_distance(w_distance); % affect new labels to the data
    
    % updating centroid value
    ao_centroid = kmeans_update_centroids(w_Data,w_Data(:,size(ai_Data,2)),ao_centroid);
          
    % calculating distortion criterion
    for i=1:size(ai_Data,1)
        ao_disto(1,w_iter) = ao_disto(1,w_iter) + norm(ai_Data(i,1:size(ai_Data,2)-1) - ao_centroid(w_Data(i,size(ai_Data,2)),:))^2;
    end
    
    % checking if the algorithm has converged
    if w_iter == 1
        convergence = 0;
    
    elseif w_iter == ai_maxIteration
        convergence = 1;
    
    elseif ao_disto(1,w_iter) == ao_disto(1,w_iter-1)
        convergence = 1;
    
    else
        convergence = 0;
    end
    w_iter = w_iter + 1;   
end
w_iter = w_iter - 1; % returning the number of iterations of kmeans until convergence
end