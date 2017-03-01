%% calculates distance between points & centroids

function ao_distances = kmeans_distance_calculation(ai_centers, ai_Data)
ao_distances = zeros(size(ai_Data,1),size(ai_centers,1));
    for k=1:size(ai_Data,1) %setting up loop through all data points
        for c=1:size(ai_centers,1) %loop through all centroids
            ao_distances(k,c) = norm(ai_centers(c,:) - ai_Data(k,1:size(ai_Data,2)-1)); %calculating l2 norm
        end
    end
end