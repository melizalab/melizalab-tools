#include <Python.h>
#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include <cstdio>
#include <vector>
#include <set>
#include <map>
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
readklu(FILE* cfp, FILE* ffp, const Py::List &atimes, map<int, vector<vector<long> > > &uvec) 
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
	int nepisodes = atimes.size();
	uvec.clear();
	//printf("Episodes: %d\n", nepisodes);
	for (set<int>::const_iterator it = clusters.begin(); it != clusters.end(); it++) {
		int cluster = *it;
		uvec[cluster] = vector<vector<long> >(nepisodes, vector<long>(0));
	}

	// now iterate through the two files concurrently
	// while matching times to the episode times
	int episode = 0;
	long et = Py::Int(atimes[episode]);
	long nt = Py::Int(atimes[episode+1]);
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
					nt = Py::Int(atimes[episode+1]);
				else
					nt = -1;
 			}
			uvec[clust][episode].push_back(atime - et);
		}
	}
	//printf("\n");

}


class _readklu_module : public Py::ExtensionModule<_readklu_module>
{
public:
	_readklu_module()
		: Py::ExtensionModule<_readklu_module>( "_readklu" )
		{
			add_keyword_method("readclusters", &_readklu_module::rk_readclusters, 
					   "readclusters (fetfile, clufile, eptimes, samplerate=20) - sorts events into episodes");
			add_varargs_method("getclusters", &_readklu_module::rk_getclusters,
					   "getclusters (clufile) - returns a list of the clusters defined in the cluters file");
                                            

			initialize( "Reads events from klusters files and assigns them to episodes" );

		}
	virtual ~_readklu_module() {}

private:

	Py::Object
	rk_getclusters(const Py::Tuple &args) {
		
		FILE *cfp;
		Py::String cname(args[0]);
		if ((cfp = fopen(cname.as_std_string().c_str(), "rt"))==NULL)
			throw Py::NameError("Could not open file " + cname.as_std_string());

		set<int> clusters;
		getclusters(cfp, clusters);

		fclose(cfp);
		Py::List out;
		for (set<int>::const_iterator it = clusters.begin(); it != clusters.end(); it++)
			out.append(Py::Int(*it));

		return out;
	}
		

	Py::Object
	rk_readclusters(const Py::Tuple &args, const Py::Dict &kws) {

		FILE *cfp, *ffp;
		if (args.size() != 3)
			throw Py::TypeError("readclusters requires 3 arguments.");

		// convert arguments to appropriate types
		Py::String fname(args[0]);
		Py::String cname(args[1]);
		Py::List atimes(args[2]);
		float samplerate;
		if (kws.hasKey("samplerate"))
			samplerate = Py::Float(kws["samplerate"]);
		else
			samplerate = 20.0;

		// open the filenames
		if ((cfp = fopen(cname.as_std_string().c_str(), "rt"))==NULL)
			throw Py::NameError("Could not open file " + cname.as_std_string());
		if ((ffp = fopen(fname.as_std_string().c_str(), "rt"))==NULL)
			throw Py::NameError("Could not open file " + fname.as_std_string());
				
  		map<int, vector<vector<long> > > uvec;
		readklu(cfp, ffp, atimes, uvec);

		fclose(cfp);
		fclose(ffp);

 		// convert output to python lists
 		Py::List ulist;
 		for (map<int, vector<vector<long> > >::const_iterator it = uvec.begin();
		     it != uvec.end(); it++) {
			int unit = it->first;
			//printf("cluster %d.\n", unit);
 			int nreps = uvec[unit].size();
 			Py::List rlist;
 			for (int j = 0; j < nreps; j++) {
 				Py::List events;
 				int nevents = uvec[unit][j].size();
 				for (int k = 0; k < nevents; k++)
 					events.append(Py::Float(uvec[unit][j][k] / samplerate));
 				rlist.append(events);
 			}
 			ulist.append(rlist);
 		}
	
		return ulist;
	}
};

extern "C" void init_readklu()
{
	static _readklu_module* _readklu = new _readklu_module;
}
