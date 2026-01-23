# kubectl

kubectl get deployments     // -n <namespace-name>
kubectl get statefulsets    // -n <namespace-name>
kubectl get daemonsets      // -n <namespace-name>
kubectl get replicasets     // -n <namespace-name>
kubectl get jobs            // -n <namespace-name>
kubectl get cronjobs        // -n <namespace-name>

## Pods
kubectl describe pods ${POD_NAME} // or --all-namespaces
kubectl get pods -n <namespace-name>
kubectl logs <pod_name>     // or -n <namespace_name> if pod is not in your current namespace
kubectl delete pod <pod-name>   // or -n <namespace-name> if pod is not in your current namespace
kubectl delete pod <pod-name> --force --grace-period=0
kubectl delete pods --all

Note: In most production scenarios, pods are managed by a controller like a Deployment or ReplicaSet. If you delete a pod that is part of a Deployment, the controller will automatically create a new replacement pod to maintain the desired replica count. If your goal is to permanently stop or remove the application, you should delete the associated Deployment instead. 

## Deployments
kubectl get deployments --all-namespaces
kubectl get deployments --namespace=<namespace-name>
kubectl get deployment <deployment-name> -o yaml
kubectl delete deployment <deployment-name>


## Namespaces
kubectl get namespaces
kubectl describe namespace <namespace-name>

## Services
kubectl get services --all-namespace