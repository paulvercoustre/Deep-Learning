%% updates the centroids

function ao_centroid = kmeans_update_centroids(ai_Data,ai_labels,ai_centroid)
    ao_centroid = zeros(size(ai_centroid)); %create a zero matrice for new centroids
    for centroid_idx=1:size(ao_centroid,1)
        [w_row] = find(ai_labels == centroid_idx); % get row coordinate in w_label of points with centroid = centroid_idx
        ao_centroid(centroid_idx,:) = mean(ai_Data(w_row,1:size(ai_Data,2)-1)); %calculates the mean coordinates      
    end
end    