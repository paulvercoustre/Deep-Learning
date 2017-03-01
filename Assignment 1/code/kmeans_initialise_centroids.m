%% randomly selects "ai_k" points from data as initial centroids

function ao_centers = kmeans_initialise_centroids(ai_K, ai_Data) % ai_K is number of centers
    ao_centers = datasample(ai_Data(:,1:size(ai_Data,2)-1),ai_K,'Replace',false);
end