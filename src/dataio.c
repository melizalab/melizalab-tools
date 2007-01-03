/*
 * dataio.c
 *
 * A python wrapper for Amish Dave's libdataio; provides IO for a variety
 * of data formats. Data are stored in numpy arrays
 *
 */

#include <Python.h>
#include <pcmio.h>
//#include <numpy/arrayobject.h>

static PyObject*
dataio(PyObject* self, PyObject* args)   /* proto for all exposed funcs */
{
	if(!PyArg_ParseTuple(args, ""))    /* receive arguments (no args) */
		return 0;                      /* propagate error if any */
	return Py_BuildValue("s","it works!"); /* build and return result */
}

/*
 * Reads in a pcm file as an array
 */
/* static PyObject* */
/* readpcm(PyObject* self, PyObject* args) */
/* { */
/* 	const int entry=1; */
/* 	char *filename; */
/* 	PCMFILE *pfp; */
/* 	struct pcmstat pstat; */
/* 	int len; */

/* 	short *buf_p; */
/* 	PyArrayObject *pcmdata; */
	
/* 	/\* parse args *\/ */
/* 	if(!PyArg_ParseTuple(args, "s", &filename))     */
/* 		return NULL; */

/* 	/\* open pcm file *\/ */
/* 	pfp = pcm_open(filename, "r"); */
/* 	pcm_stat(pfp, &pstat); */

/* 	if (pcm_seek(pfp, entry) == -1) { */
/* 		pcm_close(pfp); */
/* 		PyErr_SetString(PyExc_IOError, "Unable to seek to entry 1 of file."); */
/* 		return NULL; */
/* 	} */

/* 	/\* allocate data *\/ */
/* 	len = pstat.nsamples; */
/* 	pcmdata = (PyArrayObject*) PyArray_FromDims(1,&len,PyArray_SHORT); */
/* 	buf_p = (short *)pcmdata->data; */
	
/* 	/\* read it in *\/ */
/* 	if (pcm_read(pfp, &buf_p, &len) == -1) { */
/* 		pcm_close(pfp); */
/* 		Py_XDECREF(pcmdata); */
/* 		PyErr_SetString(PyExc_IOError, "Unable to read from file."); */
/* 		return NULL; */
/* 	} */
	
/* 	//ret = PyBuildValue("O", pcmdata); */
/* 	pcm_close(pfp); */
/* 	//Py_XDECREF(pcmdata); */
/* 	return PyArray_Return(pcmdata); */
/* } */
	
static PyObject*
getentries(PyObject* self, PyObject* args)
{
	char *filename;
	PCMFILE *pfp;

	/* parse args */
	if(!PyArg_ParseTuple(args, "s", &filename))    
		return NULL;
	
	if ((pfp = pcm_open(filename, "r")) == NULL) {
		PyErr_SetString(PyExc_IOError, "Unable to open file.");
		return NULL;
	}
	return Py_BuildValue("i", pfp->nentries);
	
}


static PyMethodDef dataioMethods[] = {   /* methods exposed from module */
	{"dataio", dataio, METH_VARARGS, "A dataio function"},    /* descriptor */
	{"getentries", getentries, METH_VARARGS, "Returns the number of entries in a pcm_seq2 file"},
	{0}                                                 /* sentinel   */
};

PyMODINIT_FUNC
initdataio()                             /* called by Python on import */
{
	Py_InitModule("dataio", dataioMethods);  /* init module with methods */
}
