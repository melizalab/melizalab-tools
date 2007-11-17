#include <Python.h>
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
readklu(FILE* cfp, FILE* ffp, const vector<long>& atimes, map<int, vector<vector<long> > > &uvec) 
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
	long et = atimes[episode];
	long nt = atimes[episode+1];
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
					nt = atimes[episode+1];
				else
					nt = -1;
 			}
			uvec[clust][episode].push_back(atime - et);
		}
	}
	//printf("\n");

}

static PyObject*
readklu_getclusters(PyObject* self, PyObject* args)
{
	const char *cname;
	if (!PyArg_ParseTuple(args, "s", &cname))
		return NULL;
	FILE *cfp;
	if ((cfp = fopen(cname, "rt"))==NULL) {
		PyErr_SetString(PyExc_IOError, "Unable to open file");
		return NULL;
	}
	
	set<int> clusters;
	getclusters(cfp, clusters);
	fclose(cfp);

	// construct a list of the clusters
	PyObject* out = PyList_New(clusters.size());
	int i = 0;
	for (set<int>::const_iterator it = clusters.begin(); it != clusters.end(); it++, i++)
		PyList_SetItem(out, i, PyInt_FromLong((long)*it));

	return out;
}

static PyObject*
readklu_readclusters(PyObject* self, PyObject* args)
{
	const char *fname, *cname;
	PyObject *abstimes, *iterator, *item;
	float samplerate = 20.0;
	vector<long> atimes;

	if (!PyArg_ParseTuple(args, "ssO|f", &fname, &cname, &abstimes, &samplerate))
		return NULL;
	iterator = PyObject_GetIter(abstimes);
	if (iterator == NULL) {
		PyErr_SetString(PyExc_TypeError, "abstimes must be a sequence");
		return NULL;
	}

	// open the files
	FILE *cfp, *ffp;
	if ((cfp = fopen(cname, "rt"))==NULL) {
		PyErr_Format(PyExc_IOError, "Unable to open file %s", cname);
		return NULL;
	}
	if ((ffp = fopen(fname, "rt"))==NULL) {
		PyErr_Format(PyExc_IOError, "Unable to open file %s", fname);
		return NULL;
	}

	// convert atimes to a vector
	while ((item = PyIter_Next(iterator))) {
		long atime = PyInt_AsLong(item);
		if (atime < 0) break;  // negative numbers are bad, or indicate non-integers
		atimes.push_back(atime);
		Py_DECREF(item);
	}
	Py_DECREF(iterator);

	if (PyErr_Occurred()) {
		PyErr_SetString(PyExc_TypeError, "Elements of abstimes must be positive integers");
		return NULL;
	}

	// run the cluster grouping function
	map<int, vector<vector<long> > > uvec;
	readklu(cfp, ffp, atimes, uvec);

	fclose(cfp);
	fclose(ffp);

	// convert output to python lists
	PyObject *ulist, *rlist, *events;
	ulist = PyList_New(0);

	for (map<int, vector<vector<long> > >::const_iterator it = uvec.begin();
	     it != uvec.end(); it++) {
		int unit = it->first;
		//printf("cluster %d.\n", unit);
		int nreps = uvec[unit].size();
		rlist = PyList_New(nreps);
		for (int j = 0; j < nreps; j++) {
			int nevents = uvec[unit][j].size();
			events = PyList_New(nevents);
			for (int k = 0; k < nevents; k++)
				PyList_SetItem(events, k, PyFloat_FromDouble(uvec[unit][j][k] / samplerate));
			PyList_SetItem(rlist, j, events);  // takes reference to events
		}
		PyList_Append(ulist, rlist);   // does not decref rlist
		Py_DECREF(rlist);
	}
	
	return ulist;
}

static PyMethodDef readklu_methods[] = {
    {"getclusters",  readklu_getclusters, METH_VARARGS,
     "Return a list of the clusters defined in the clu file."},
    {"readclusters",  readklu_readclusters, METH_VARARGS,
     "Return the cluster assignments of all the events in the clu file which are also in abstimes."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

extern "C" {
PyMODINIT_FUNC
init_readklu(void)
{
    (void) Py_InitModule("_readklu", readklu_methods);
}
}
