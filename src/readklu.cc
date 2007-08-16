#include<cstdio>
#include<vector>

using std::vector;

int allclust = 0;
char cname[] = "site_4_2.clu.1";
char fname[] = "site_4_2.fet.1";
long _atimes[] = {76134603, 76194603, 76254603, 76314603, 76374603, 76434603,
		  76494603, 76554603, 76614603, 76674603, 76734603, 76794603};
int _natimes = 12;

int main(int argc, char *argv[]) {

	FILE* cfp = fopen(cname, "rt");
	FILE* ffp = fopen(fname, "rt");
	int rp = 0;
	int nclusts, nfeats;
	int startclust = 0;
	int nlines = 0;

	// initialize atimes
	vector<long> atimes;
	for (int i = 0; i < _natimes; i++) {
		atimes.push_back((long)_atimes[i]);
	}


	// number of clusters and features
	rp = fscanf(cfp,"%d\n", &nclusts);
	rp = fscanf(ffp, "%d\n", &nfeats);
	
	// because we're dropping 0 and 1 (but only if there are more than 1 or 2
	// units) we have to scan through the cluster file to see if they're there
	bool has_zero = false;
	bool has_one = false;
	int clust;
	while(rp != EOF) {
		rp = fscanf(cfp, "%d\n", &clust);
		if (!has_zero && clust==0)
			has_zero = true;
		if (!has_one && clust==1)
			has_one = true;
		nlines +=1;
	}
	fseek(cfp, 0, 0);
	rp = fscanf(cfp, "%d\n", &nclusts);
	// with one cluster, we start at 0 (and end there)
	// with two clusters, we drop 0 or 1 (whichever is lowest)
	// with three clusters or more, we drop 0 and 1, if either is present
	if (nclusts==2 && (has_zero || has_one))
		startclust = 1;
	if (nclusts>2 && has_zero)
		startclust = 1;
	if (nclusts>2 && has_one)
		startclust += 1;

	printf("Total events: %d\n", nlines);
	printf("Number of clusters: %d\n", nclusts);
	printf("Number of valid clusters: %d\n", nclusts-startclust);

	// allocate storage
	vector<vector<vector<long> > > uvec(nclusts-startclust, 
					    vector<vector<long> >(atimes.size(), 
							 vector<long>(0)));

	// now iterate through the two files concurrently
	vector<long>::const_iterator episodetime = atimes.begin();
	int entry = 0;
	printf("Episode 0 ");
	long atime;
	while (rp != EOF) {
 		for (int j = 0; j < nfeats && rp != EOF; j++)
 			rp = fscanf(ffp, "%ld", &atime);
		rp = fscanf(cfp, "%d\n", &clust);
		if (clust >= startclust) {
			// THIS CODE ASSUMES THE ABSTIMES ARE SORTED
 			if (atime < *episodetime)  
 				printf("\nWarning: %d comes before the current episode\n", atime);
			// Advance the pointers until the episodetime is correct
			while ((episodetime != (atimes.end()-1)) && (atime >= *(episodetime+1))) {
				entry += 1;
				episodetime++;
			}
			uvec[clust-startclust][entry].push_back(atime - *episodetime);
		}
	}
	printf("\n");
	fclose(cfp);
	fclose(ffp);
	
	for (int i = 0; i < uvec.size(); i++) {
		printf("Unit %d: [", i);
		vector<vector<long> > &rvec = uvec[i];
		for (int j = 0; j < rvec.size(); j++) 
			printf("%d ", rvec[j].size());
		printf("]\n");
	}


}


	
