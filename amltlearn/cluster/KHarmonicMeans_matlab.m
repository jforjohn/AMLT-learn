function [final_centroids, total_validations] = harmonicKmeans(...
            instances, num_clusters, num_retries,...
            max_num_iterations, error_threshold, method_validation, r)
    
    total_validations = zeros(1,length(method_validation));
    best_validation = Inf;
    for i=1:num_retries+1
        centroids = getInitialCentroids(instances, num_clusters);
        
        iteration = 0;
        difference_performance = Inf;
        performance = Inf;
        while difference_performance > error_threshold && iteration < max_num_iterations
           
            old_performance = performance;
            centroids = getNewCentroidsHKM(instances,centroids);
            performance = evaluatePerformance(instances, centroids);
            difference_performance = abs(performance-old_performance);
            
            iteration = iteration + 1;
        end
        
        prov_validations = zeros(1,3);
        if any(method_validation==1)
            classified_instances = getClassifiedInstances(instances,centroids,r);
            prov_validations(1) = validateDBI(classified_instances, centroids);
        end

        if any(method_validation==2) || any(method_validation==3)
            classified_indexes = getClassifiedIndexes(instances,centroids,2);
            if any(method_validation==2)
                prov = evalclusters(instances,classified_indexes,'CalinskiHarabasz');
                prov_validations(2) = prov.CriterionValues;
            end
            if any(method_validation==3)
                prov_validations(3) = validateDunns(num_clusters,instances,classified_indexes);
            end
        end
        
        total_validations = total_validations + prov_validations(method_validation);
        if prov_validations(1) < best_validation
            final_centroids = centroids;
            best_validation = prov_validations(1);
        end
    end
    total_validations = total_validations / (num_retries + 1);
end

##################################################

function new_centroids = getNewCentroidsHKM(instances,centroids) 
    num_clusters = size(centroids, 1);
    num_instances = size(instances,1);
    num_attributes = size(instances,2);
    new_centroids = zeros(num_clusters,num_attributes);
    
    q = zeros(num_instances, num_clusters);
    p = zeros(num_instances, num_clusters);
    for i = 1:num_instances
        d = zeros(1, num_clusters);
        instance = instances(i,:);
        for k = 1:num_clusters
            centroid = centroids(k,:);
            d(k) = max(sqrt(sum((instance-centroid).^2)),0.0001);
        end
        d_min = min(d);
        
        for k = 1:num_clusters
            if d_min ~= 0
                aux = d_min/d(k);
                aux = aux^4;

                aux2 = 0;
                for l=1:num_clusters
                    if d(l)~=d_min
                        aux2 = aux2+(d_min/d(l))^2;
                    end
                end
                aux2 = 1 + aux2;
                aux2 = aux2^2;

                q(i,k) = aux / aux2;
            else
                if d(k) == d_min
                    q(i,k) = 1;
                else
                    q(i,k) = 0;
                end
            end
        end
    end
    
    q_k = sum(q);
    for i = 1:num_instances
        p(i,:) = q(i,:) ./ q_k;
    end
    
    for k = 1:num_clusters
        for i = 1:num_instances
            aux = p(i,k)*instances(i,:);
            new_centroids(k,:) = new_centroids(k,:)+aux;
        end
    end
end

##################################################

function performance = evaluatePerformance(instances, centroids)
    num_instances = size(instances,1);
    num_clusters = size(centroids,1);
    performance = 0;
    for i = 1:num_instances
        instance = instances(i,:);
        prov_sum = 0;
        for l = 1:num_clusters
             centroid = centroids(l,:);
             prov_val = sqrt(sum((centroid-instance).^2));
             prov_sum = prov_sum + 1/(prov_val^2);
        end
        performance = performance+num_clusters/prov_sum;
    end
end




