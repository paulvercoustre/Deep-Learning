%% finds the centroid with smallest euclidian distance

function w_label = kmeans_find_minimal_distance(ai_distances)
    [~,w_label] = min(ai_distances,[],2);
end
