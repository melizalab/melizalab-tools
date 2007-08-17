#include <Python.h>
#include "numpy/arrayobject.h"
#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include <cstdio>
#include <vector>
using std::vector;

static vector<vector<vector<long> > >
readklu(FILE* cfp, FILE* ffp, const Py::List &atimes) 
{

	int rp = 0;
	int nclusts, nfeats;
	int startclust = 0;
	long nlines = 0;

	// number of clusters and features
	rp = fscanf(cfp,"%d\n", &nclusts);
	rp = fscanf(ffp, "%d\n", &nfeats);
	printf("Clusters: %d\nFeatures:%d\n", nclusts, nfeats);
	
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

	printf("Events: %ld\n", nlines);
	printf("Valid clusters: %d\n", nclusts-startclust);
	// allocate storage
	int nepisodes = atimes.size();
	printf("Episodes: %d\n", nepisodes);
        vector<vector<vector<long> > > uvec(nclusts-startclust, 
                                            vector<vector<long> >(nepisodes, vector<long>(0)));

	// now iterate through the two files concurrently
	// while matching times to the episode times
	int episode = 0;
	long et = Py::Int(atimes[episode]);
	long nt = Py::Int(atimes[episode+1]);
	long atime;
	printf("Start time: %ld\n", et);
	printf("Episode %d: ", episode);
	while (rp != EOF) {
 		for (int j = 0; j < nfeats && rp != EOF; j++)
 			rp = fscanf(ffp, "%ld", &atime);
		rp = fscanf(cfp, "%d\n", &clust);
		if (clust >= startclust) {
			// THIS CODE ASSUMES THE ABSTIMES ARE SORTED
 			if (atime < et)
 				printf("\nWarning: %ld comes before the current episode\n", atime);
			// Advance the pointers until the episodetime is correct
 			while ((nt>0) && (atime >= nt)) {
 				episode += 1;
				printf("\nEpisode %d: ", episode);
 				et = nt;
				if (episode+1 < nepisodes) 
					nt = Py::Int(atimes[episode+1]);
				else
					nt = -1;
 			}
			printf("*");
			uvec[clust-startclust][episode].push_back(atime - et);
		}
	}

	return uvec;

}


class _readklu_module : public Py::ExtensionModule<_readklu_module>
{
public:
	_readklu_module()
		: Py::ExtensionModule<_readklu_module>( "_readklu" )
		{
			add_keyword_method("readclusters", &_readklu_module::rk_readclusters, 
					   "readclusters (fetfile, clufile, eptimes, samplerate=20) - sorts events into episodes");
                                            

			initialize( "Reads events from klusters files and assigns them to episodes" );

		}
	virtual ~_readklu_module() {}

private:

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
				
  		vector<vector<vector<long> > > uvec = readklu(cfp, ffp, atimes);

		fclose(cfp);
		fclose(ffp);

 		// convert output to python lists
 		int nunits = uvec.size();
 		Py::List ulist;
 		for (int i = 0; i < nunits; i++) {
 			int nreps = uvec[i].size();
 			Py::List rlist;
 			for (int j = 0; j < nreps; j++) {
 				Py::List events;
 				int nevents = uvec[i][j].size();
 				for (int k = 0; k < nevents; k++)
 					events.append(Py::Float(uvec[i][j][k] / samplerate));
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
