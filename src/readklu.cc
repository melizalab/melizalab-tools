#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/args.hpp>
#include <cstdio>
#include <vector>
#include <set>
#include <map>
using namespace boost::python;
using std::vector;
using std::set;
using std::map;

static long
getclusters(FILE* cfp, set<int> &clusters)
{
	int clust, rp, fpos;
	long nlines = 0;
	fpos = ftell(cfp);
	fseek(cfp, 0, 0);
	rp = fscanf(cfp, "%d\n", &clust);  // throw away first line
	clusters.clear();
	while(rp != EOF) {
		rp = fscanf(cfp, "%d\n", &clust);
		clusters.insert(clust);
		nlines += 1;
	}
	fseek(cfp, fpos, 0);

	return nlines;
}

static void
readklu(FILE* cfp, FILE* ffp, const list& atimes, map<int, vector<vector<long> > > &uvec) 
{

	int rp = 0;
	int nclusts, nfeats;

	// number of clusters and features
	rp = fscanf(cfp,"%d\n", &nclusts);
	rp = fscanf(ffp, "%d\n", &nfeats);
	//printf("Clusters: %d\nFeatures: %d\n", nclusts, nfeats);
	
	// cluster numbers can be noncontiguous so we need to scan the cluster file
	int clust;
	set<int> clusters;
	getclusters(cfp, clusters);

	// with one cluster, that's the one we use
	// with more than one cluster, we drop 0
	// if there's still more than one cluster, we drop 1
	if ((clusters.size()> 1) && (clusters.count(0)))
		clusters.erase(0);
	if ((clusters.size()> 1) && (clusters.count(1)))
		clusters.erase(1);

	//printf("Events: %ld\n", nlines);
	//printf("Valid clusters: %d\n", (int)clusters.size());
	// allocate storage
	int nepisodes = len(atimes);
	uvec.clear();
	//printf("Episodes: %d\n", nepisodes);
	for (set<int>::const_iterator it = clusters.begin(); it != clusters.end(); it++) {
		int cluster = *it;
		uvec[cluster] = vector<vector<long> >(nepisodes, vector<long>(0));
	}

	// now iterate through the two files concurrently
	// while matching times to the episode times
	int episode = 0;
	long et = extract<long>(atimes[episode]);
	long nt = extract<long>(atimes[episode+1]);
	long atime;
	//printf("Start time: %ld\n", et);
	//printf("Episode %d: ", episode);
	while (rp != EOF) {
 		for (int j = 0; j < nfeats && rp != EOF; j++)
 			rp = fscanf(ffp, "%ld", &atime);
		rp = fscanf(cfp, "%d\n", &clust);
		//printf("%d", clust);
		if (clusters.count(clust)) {
			////printf("%d\n", clustind);
			// THIS CODE ASSUMES THE ABSTIMES ARE SORTED
 			if (atime < et)
 				printf("\nWarning: %ld comes before the current episode\n", atime);
			// Advance the pointers until the episodetime is correct
 			while ((nt>0) && (atime >= nt)) {
 				episode += 1;
				//printf("\nEpisode %d: ", episode);
 				et = nt;
				if (episode+1 < nepisodes) 
					nt = extract<long>(atimes[episode+1]);
				else
					nt = -1;
 			}
			uvec[clust][episode].push_back(atime - et);
		}
	}
	//printf("\n");

}

// a wrapper for getclusters
list rk_getclusters(const std::string& cname) 
{
	
	FILE *cfp;
	if ((cfp = fopen(cname.c_str(), "rt"))==NULL)
		throw std::ios_base::failure("Could not open file " + cname);

	set<int> clusters;
	getclusters(cfp, clusters);

	fclose(cfp);
	list out;
	for (set<int>::const_iterator it = clusters.begin(); it != clusters.end(); it++)
		out.append(*it);

	return out;
}

 
list rk_readclusters(const std::string& fname, const std::string& cname,
		     const list& atimes, float samplerate=20.0)
{
	
	FILE *cfp, *ffp;
	
	// open the filenames
	if ((cfp = fopen(cname.c_str(), "rt"))==NULL)
		throw std::ios_base::failure("Could not open file " + cname);
	if ((ffp = fopen(fname.c_str(), "rt"))==NULL)
		throw std::ios_base::failure("Could not open file " + fname);
				
	map<int, vector<vector<long> > > uvec;
	readklu(cfp, ffp, atimes, uvec);

	fclose(cfp);
	fclose(ffp);

	// convert output to python lists
	list ulist;
	for (map<int, vector<vector<long> > >::const_iterator it = uvec.begin();
	     it != uvec.end(); it++) {
		int unit = it->first;
		//printf("cluster %d.\n", unit);
		int nreps = uvec[unit].size();
		list rlist;
		for (int j = 0; j < nreps; j++) {
			list events;
			int nevents = uvec[unit][j].size();
			for (int k = 0; k < nevents; k++)
				events.append(uvec[unit][j][k] / samplerate);
			rlist.append(events);
		}
		ulist.append(rlist);
	}
	
	return ulist;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(rk_readclusters_overloads, rk_readclusters, 3, 4)

BOOST_PYTHON_MODULE(_readklu)
{
	def("getclusters", rk_getclusters);
	def("readclusters", rk_readclusters, rk_readclusters_overloads(
		    args("samplerate")));
}
		
