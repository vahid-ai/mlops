# Kubectl
kubectl logs <pod-name>	// View a snapshot of logs from the main container in a pod

kubectl logs -f <pod-name>	// Stream (follow) logs in real-time (use Ctrl + C to exit)

kubectl logs -c <container-name> <pod-name>	// View logs from a specific container within a multi-container pod

kubectl logs --previous <pod-name>	// View logs from a previously terminated container instance (useful for crash loops)

kubectl logs --tail=<number> <pod-name>	// Display only the most recent N lines of output (e.g., --tail=50)

kubectl logs --since=<duration> <pod-name>	// Show logs from only the last specified duration (e.g., --since=1h, --since=30m)

kubectl logs <pod-name> > <file-name.log>	// Save logs to a local file for later analysis

kubectl logs -l app=my-app	Retrieve logs from all pods matching a specific label

kubectl get pods

kubectl get pods --all-namespaces

kubectl get pods --namespace <namespace_name> (or -n <namespace_name>)

kubectl config get-clusters

# Project Specific
- Reason: Need to verify the Spark operator pod is now Running after updating the image tag - `kubectl -n dfp get pods -l app=spark-operator -o wide`

# References
1. https://grafana.com/go/webinar/kubernetes-monitoring-with-grafana-cloud/?src=ggl-s&mdm=cpc&camp=nb-kubernetes-exact-amer&cnt=148128497448&trm=kubernetes%20logs&device=c&gad_source=1&gad_campaignid=18630997354&gbraid=0AAAAADkOfqvRvAmVg3MvnQzBP7EAF0tbw&gclid=EAIaIQobChMIi6WHopS-kQMVCAKtBh2cMgUMEAAYASAAEgITbvD_BwE
2. https://spacelift.io/blog/kubectl-logs
3. https://kubernetes.io/docs/reference/kubectl/generated/kubectl_logs/
4. https://jamesdefabia.github.io/docs/user-guide/kubectl/kubectl_logs/
5. https://docs.spacelift.io/concepts/worker-pools/kubernetes-workers