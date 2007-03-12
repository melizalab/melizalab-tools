
/* Copyright (C) 2001 by Amish S. Dave (adave3@uic.edu) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include "pcmio.h"

int pcm_errno;

typedef struct
{
	int (*recognizer)(PCMFILE *);
	int (*open)(PCMFILE *);
} PCMTYPE;

PCMTYPE pcm_types[] =
{
	{pcmseq_recognizer, pcmseq_open},
	{NULL, NULL}
};

PCMFILE *pcm_open(char *filename, char *type)
{
	PCMFILE *newfp;
	int typefound, i;

	/*
	** Validate the arguments
	*/
	if ((type == NULL) || ((type[0] != 'r') && (type[0] != 'w')))
	{
		errno = EINVAL;
		return NULL;
	}
	if ((type[0] == 'r') && (access(filename, R_OK|F_OK) == -1))
		return NULL;

	pcm_errno = 0;
	if ((newfp = (PCMFILE *)malloc(sizeof(PCMFILE))) != NULL)
	{
		if ((newfp->name = (char *)malloc(strlen(filename) + 1)) != NULL)
		{
			strcpy(newfp->name, filename);
			newfp->flags = (type[0] == 'w') ? (O_WRONLY) : (O_RDONLY);
			newfp->close = NULL;
			newfp->read = NULL;
			newfp->write = NULL;
			newfp->seek = NULL;
			newfp->ctl = NULL;
			newfp->stat = NULL;
			newfp->p2file = NULL;
			newfp->tempnam = NULL;
			newfp->entry = 1;
			/*
			** Identify the TYPE of pcm file
			*/
			typefound = 0;
			for (i=0; pcm_types[i].recognizer != NULL; i++)
				if ((typefound = (*(pcm_types[i].recognizer))(newfp)) != -1)
					break;
			if (typefound != -1)
			{
				/*
				** Call the type-specific open function
				*/
				newfp->open = pcm_types[i].open;
				if ((*(newfp->open))(newfp) != -1)
					return newfp;
			}
			else pcm_errno = EINVAL;
			free(newfp->name);
		}
		else pcm_errno = ENOMEM;
		free(newfp);
	}
	else pcm_errno = ENOMEM;
	errno = pcm_errno;
	return NULL;
}

void pcm_close(PCMFILE *fp)
{
	/*
	** Call the type-specific close function
	*/
	if (fp->close != NULL)
		(*(fp->close))(fp);

	/*
	** Free the storage allocated by the generic open function
	*/
	free(fp->name);
	free(fp);
}

int pcm_read(PCMFILE *fp, short **buf_p, int *nsamples_p)
{
	/*
	** Validate arguments
	*/
	if ((buf_p != NULL) && (nsamples_p != NULL))
	{
		if (fp->read != NULL)
		{
			/*
			** Call the type-specific read function
			*/
			return (*(fp->read))(fp, buf_p, nsamples_p);
		}
	}
	errno = EINVAL;
	return -1;
}

int pcm_seek(PCMFILE *fp, int entry)
{
	/*
	** Call the type-specific seek function
	*/
	if (fp->seek != NULL)
		return (*(fp->seek))(fp, entry);
	errno = EINVAL;
	return -1;
}

/*
** request types:
**  PCMIOMALLOC
**  PCMIOMMAP
** 		Use to set preference for MALLOC or MMAP allocation methods...
**		The argument is ignored.
*/
int pcm_ctl(PCMFILE *fp, int request, void *arg)
{
	if (fp->ctl != NULL)
		return (*(fp->ctl))(fp, request, arg);
	errno = EINVAL;
	return -1;
}

int pcm_stat(PCMFILE *fp, struct pcmstat *buf)
{
	if (fp->stat != NULL)
		return (*(fp->stat))(fp, buf);
	errno = EINVAL;
	return -1;
}

int pcm_write(PCMFILE *fp, short *buf, int nsamples)
{
	if (fp->write != NULL)
		return (*(fp->write))(fp, buf, nsamples);
	errno = EINVAL;
	return -1;
}

// from pcmseq2.c

/*
** Return 0 if the file is the file is of this type; return -1 otherwise
*/
int pcmseq_recognizer(PCMFILE *fp)
{
	if ((strncasecmp(fp->name + strlen(fp->name) - 9, ".pcm_seq2", 9)) &&
		(strncasecmp(fp->name + strlen(fp->name) - 8, ".pcm_seq", 8)) &&
		(strncasecmp(fp->name + strlen(fp->name) - 8, ".pcmseq2", 8)) &&
		(strncasecmp(fp->name + strlen(fp->name) - 7, ".pcmseq", 7)))
		return -1;
	return 0;
}

int pcmseq_open(PCMFILE *fp)
{
	fp->memalloctype = PCMIOMMAP;
	fp->tempnam = NULL;
	fp->close = pcmseq_close;
	fp->read = pcmseq_read;
	fp->write = pcmseq_write;
	fp->seek = pcmseq_seek;
	fp->ctl = pcmseq_ctl;
	fp->stat = pcmseq_stat;
	fp->addr = NULL;
	fp->len = 0;
	fp->bufptr = NULL;
	fp->buflen = 0;
	fp->samplerate = 20000;
	fp->fd = -1;
	fp->outfp = NULL;
	fp->timestamp = 0;
	fp->microtimestamp = 0;
	fp->nentries = 0;
	fp->entry = 1;
	if (fp->flags == O_RDONLY)
	{
		if ((fp->p2file = pcmseq2_open(fp->name)) == NULL)
		{
			pcm_errno = EIO;
			return -1;
		}
		pcmseq2_getinfo(fp->p2file, 1, NULL, &(fp->samplerate), NULL, &(fp->nentries));
	}
	if (fp->flags == O_WRONLY)
	{
		fp->entrystarted = 0;
		if ((fp->outfp = fopen(fp->name, "w")) == NULL)
		{
			pcm_errno = errno;
			return -1;
		}
	}
	return 0;
}

void pcmseq_close(PCMFILE *fp)
{
	if (fp->flags == O_RDONLY)
	{
		/* Unmap the mapped file, OR free the malloc()ed memory */
		if (fp->addr != NULL)
		{
			if (fp->memalloctype == PCMIOMMAP)
				munmap(fp->addr, fp->len);
			else if (fp->memalloctype == PCMIOMALLOC)
				free(fp->addr);
		}
		/* Clean up after the temporary file */
		if (fp->tempnam != NULL)
		{
			close(fp->fd);
			free(fp->tempnam);
			fp->tempnam = NULL;
		}
		/* Close and free the pcmseq2-reading file structures */
		if (fp->p2file != NULL)
			pcmseq2_close(fp->p2file);
	}
	if (fp->flags == O_WRONLY)
	{
		/* Mark current entry finished. */
		if (fp->entrystarted != 0) pcmseq2_write_data(fp, NULL, 0, 1);
		if (fp->outfp != NULL) fclose(fp->outfp);
	}
	return;
}

int pcmseq_read(PCMFILE *fp, short **buf_p, int *nsamples_p)
{
	long numwritten, entrysize;
	void *addr;
	int len;
	int samplerate;

	if (fp->flags == O_RDONLY)
	{
		/*
		** Clean up from previous calls
		*/
		if (fp->addr != NULL)
		{
			if (fp->memalloctype == PCMIOMMAP)
				munmap(fp->addr, fp->len);
			else if (fp->memalloctype == PCMIOMALLOC)
				free(fp->addr);
		}
		fp->addr = NULL;
		fp->bufptr = NULL;
		fp->len = fp->buflen = 0;

		/*
		** Get the size
		*/
		if (pcmseq2_getinfo(fp->p2file, fp->entry, &entrysize, &samplerate, NULL, NULL) != 0)
		{
			pcm_errno = EIO;
			return -1;
		}
		len = entrysize * 2;
		fp->samplerate = samplerate;

		/*
		** Either mmap the tmp file (extending it to the proper size)
		** or malloc the needed space.
		*/
		if (fp->memalloctype == PCMIOMMAP)
		{
			/*
			** Open a temporary file; each call to read() reads an entry into this file;
			** the file is then mmap'ed (in read()). seek() determines which entry will be read next.
			** NOTE: by unlinking this file immediately after it opens, I ensure that it gets
			** deleted after this program quits (even if it crashes).  It doesn't affect the usage
			** of the file while it is kept open, however.
			** The location of the temporary file can be modified by the environment variable TMPDIR.
			** NOTE: it can be used multiple times, if pcmseq_read() is called again...
			*/
			if (fp->tempnam == NULL)
			{
				fp->tempnam = strdup("/tmp/pcmXXXXXX");
				if ((fp->fd = mkstemp(fp->tempnam)) == -1)
				{
					free(fp->tempnam);
					fp->tempnam = NULL;
					pcm_errno = errno;
					return -1;
				}
				unlink(fp->tempnam);
			}
			ftruncate(fp->fd, len);
			lseek(fp->fd, 0, SEEK_SET);
			addr = (void *)mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_PRIVATE, fp->fd, 0);
			if (addr == (void *)-1)
			{
				pcm_errno = errno;
				return -1;
			}
			madvise(addr, len, MADV_SEQUENTIAL);
		}
		else if (fp->memalloctype == PCMIOMALLOC)
		{
			if ((addr = (void *)malloc(len)) == NULL)
			{
				pcm_errno = EINVAL;
				return -1;
			}
		}
		else return -1;

		/*
		** read into the tmp file
		*/
		if ((pcmseq2_read(fp->p2file, fp->entry, 0, len/2 - 1, addr, &numwritten, &entrysize) != 0) ||
			((numwritten != entrysize) && (numwritten != (len/2))))
		{
			pcm_errno = EIO;
			if (fp->memalloctype == PCMIOMMAP)
				munmap(addr, len);
			else if (fp->memalloctype == PCMIOMALLOC)
				free(addr);
			return -1;
		}
		if (fp->memalloctype == PCMIOMMAP)
			mprotect(addr, len, PROT_READ);

		/*
		** We were successful
		*/
		fp->addr = addr;
		fp->len = len;
		fp->bufptr = addr;
		fp->buflen = fp->len;
		*buf_p = fp->bufptr;
		*nsamples_p = fp->buflen / 2;
		return 0;
	}
	pcm_errno = EOPNOTSUPP;
	return -1;
}

int pcmseq_seek(PCMFILE *fp, int entry)
{
	if (fp->flags == O_WRONLY)
	{
		if (fp->outfp != NULL)
		{
			if (fp->entrystarted != 0)
			{
				/*
				** Mark current entry finished; clear entrystarted so pcmseq_write() can write a new header.
				*/
				pcmseq2_write_data(fp, NULL, 0, 1);
				fp->entrystarted = 0;
			}
			fp->entry = entry;
			return 0;
		}
	}
	if (fp->flags == O_RDONLY)
	{
		if ((entry > 0) && (entry <= fp->nentries))
		{
			fp->entry = entry;
			return 0;
		}
	}
	pcm_errno = EINVAL;
	return -1;
}

int pcmseq_ctl(PCMFILE *fp, int request, void *arg)
{
	struct pcmstat buf;

	if (fp->flags == O_RDONLY)
	{
		/* Only allow a change if we haven't already read the file. */
		if ((fp->addr == NULL) && ((request == PCMIOMALLOC) || (request == PCMIOMMAP)))
		{
			fp->memalloctype = request;
			return 0;
		}
		if ((request == PCMIOGETSIZE) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(int *)arg = buf.nsamples;
			return 0;
		}
		if ((request == PCMIOGETSR) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(int *)arg = buf.samplerate;
			return 0;
		}
		if ((request == PCMIOGETENTRY) && (arg != NULL))
		{
			*(int *)arg = fp->entry;
			return 0;
		}
		if ((request == PCMIOGETNENTRIES) && (arg != NULL))
		{
			*(int *)arg = fp->nentries;
			return 0;
		}
		if ((request == PCMIOGETTIME) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(int *)arg = buf.timestamp;
			return 0;
		}
		if ((request == PCMIOGETTIMEFRACTION) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(long *)arg = buf.microtimestamp;
			return 0;
		}
	}
	if (fp->flags == O_WRONLY)
	{
		if ((request == PCMIOGETSIZE) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(int *)arg = buf.nsamples;
			return 0;
		}
		if ((request == PCMIOGETCAPS) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(int *)arg = buf.capabilities;
			return 0;
		}
		if ((request == PCMIOGETSR) && (arg != NULL))
		{
			pcmseq_stat(fp, &buf);
			*(int *)arg = buf.samplerate;
			return 0;
		}
		if ((request == PCMIOGETENTRY) && (arg != NULL))
		{
			*(int *)arg = fp->entry;
			return 0;
		}
		if ((request == PCMIOGETNENTRIES) && (arg != NULL))
		{
			*(int *)arg = fp->nentries;
			return 0;
		}
		if ((request == PCMIOSETSR) && (arg != NULL))
		{
			fp->samplerate = *(int *)arg;
			return 0;
		}
		if ((request == PCMIOSETTIME) && (arg != NULL))
		{
			fp->timestamp = *(int *)arg;
			return 0;
		}
		if ((request == PCMIOSETTIMEFRACTION) && (arg != NULL))
		{
			fp->microtimestamp = *(long *)arg;
			return 0;
		}
	}
	pcm_errno = EINVAL;
	return -1;
}

int pcmseq_stat(PCMFILE *fp, struct pcmstat *buf)
{
	long entrysize;
	int samplerate, nentries;
	unsigned long long datetime;

	if (fp->flags == O_RDONLY)
	{
		if (fp->p2file != NULL)
		{
			if (pcmseq2_getinfo(fp->p2file, fp->entry, &entrysize, &samplerate, &datetime, &nentries) != 0)
			{
				pcm_errno = EIO;
				return -1;
			}
			buf->entry = fp->entry;
			buf->nentries = nentries;
			buf->nsamples = entrysize;
			buf->samplerate = samplerate;
			buf->timestamp = ((datetime - 0x007c95674beb4000LL) / 10000000 + 18000);
			buf->microtimestamp = ((datetime - 0x007c95674beb4000LL) % 10000000 / 10);
			buf->capabilities = PCMIOCAP_MULTENTRY | PCMIOCAP_SAMPRATE;
			return 0;
		}
	}
	if (fp->flags == O_WRONLY)
	{
		buf->entry = fp->entry;
		buf->nsamples = fp->pcmseq3_entrysize;
		buf->samplerate = fp->samplerate;
		buf->capabilities = PCMIOCAP_MULTENTRY | PCMIOCAP_SAMPRATE;
		return 0;
	}
	pcm_errno = EINVAL;
	return -1;
}


/*
** This can be called repetitively; it will keep appending to the current entry.
** To finish the entry, call pcmseq_seek() with the next entry #.
** To finish the entry AND file, call pcmseq_close()
*/
int pcmseq_write(PCMFILE *fp, short *buf, int nsamples)
{
	int rc;

	rc = -1;
	if (fp->flags == O_WRONLY)
	{
		rc = 0;
		if (fp->outfp != NULL)
		{
			if (fp->entrystarted == 0)
			{
				rc = pcmseq2_write_hdr(fp);
				fp->entrystarted = 1;
			}
			if (rc == 0) rc = pcmseq2_write_data(fp, buf, nsamples, 0);
			return rc;
		}
	}
	pcm_errno = EINVAL;
	return rc;
}


// code from pcmseq2_read.c
// now the pcm_seq2 proper code
//#if __BYTE_ORDER == __BIG_ENDIAN
#if __BIG_ENDIAN__
#define sexconv16(in) ((((in) & 0x00ff) << 8 ) | (( (in) & 0xff00) >> 8))
#define sexconv32(in) ( (((in) & 0x000000ff) << 24 ) | (( (in) & 0x0000ff00) << 8) | (( (in) & 0x00ff0000) >> 8) | (( (in) & 0xff000000) >> 24) )
#define sexconv16_array(buf, cnt) \
	{\
		int n;\
		short *bufptr = (buf);\
		for (n=0; n < (cnt); n++)\
			bufptr[n] = sexconv16(bufptr[n]);\
	}
#define sexconv_needed
#else
#define sexconv16(in) (in)
#define sexconv32(in) (in)
#define sexconv16_array(buf, cnt)
#endif

static void skipsegment(P2FILE *p2fp);
static int readentryhdr(P2FILE *p2fp);
static int readsegment(P2FILE *p2fp);
static int peekcontrolword(P2FILE *p2fp, unsigned short *controlword_p);
static int pcmseq2_indexfile(P2FILE *p2fp);

static int pcmseq2_scantoentrystart(FILE *fp, int type, int segmentsize, int entryhdrsize);
static int scantosegmentstart(P2FILE *p2fp);

P2FILE *pcmseq2_open(char *filename)
{
	P2FILE *p2fp = NULL;
	int i;

	if ((p2fp = (P2FILE *)calloc(sizeof(P2FILE), 1)) != NULL)
	{
		p2fp->lastentry = 0;
		if ((p2fp->fp = fopen(filename, "r")) != NULL)
		{
			p2fp->currententry = 1;
			for (i=0; i < CACHESIZE; i++)
			{
				p2fp->entry_pos_cache[i] = -1;
				p2fp->entry_size_cache[i] = -1;
				p2fp->entry_sr_cache[i] = 0;
				p2fp->entry_time_cache[i] = 0;
			}
			if (pcmseq2_indexfile(p2fp) == 0)
			{
				rewind(p2fp->fp);
				return p2fp;
			}
			fclose(p2fp->fp);
		}
		free(p2fp);
	}
	return NULL;
}


void pcmseq2_close(P2FILE *p2fp)
{
	if (p2fp != NULL)
	{
		(void)fclose(p2fp->fp);
		free(p2fp);
	}
}

int pcmseq2_read(P2FILE *p2fp, int entry, long start, long stop, short *buf, long *numwritten_p, long *entrysize_p)
{
	FILE *fp;
	long status;
	int curentry;
	unsigned short controlword;
	long *cache, curpos, startpad, endpad, blockstart, blockstop, startpos, recordwords = 0;
	short *samples;

	if (numwritten_p != NULL) *numwritten_p = 0;
	if (entrysize_p != NULL) *entrysize_p = 0;

	/*
	** Get to the beginning of the entry we want to read
	*/
	if (pcmseq2_seektoentry(p2fp, entry) == -1)
		return -1;

	/*
	** Store some convenience variables
	*/
	fp = p2fp->fp;
	curentry = p2fp->currententry;
	cache = p2fp->entry_pos_cache;

	/*
	** Cache the current position.  If there is an error, we will come back here, rather than
	** leaving ourselves in some undefined position in the middle of an entry.
	*/
	startpos = ftell(fp);

	/*
	** Read and validate the entry header
	*/
	if (readentryhdr(p2fp) == -1)
	{
		fseek(fp, startpos, SEEK_SET);
		return -1;
	}

	status = -1;
	blockstart = 0;
	curpos = 0;
	for (;;)
	{
		if (peekcontrolword(p2fp, &controlword) == -1)
			break;

		/* We are either at an entry header, or at a segment header */
		if (controlword == 0x03)							/* ENTRY */
		{
			curentry++;
			cache[curentry] = ftell(fp);
			status = 0;
			recordwords = p2fp->p2seg.recordwords;
			break;										/* We've finished this entry */
		}
		else if (controlword == 0x01)						/* SEGMENT */
		{
			/*
			** If we want any of this data, then we read the whole segment.  Otherwise, we can
			** just seek past it
			*/
			blockstop = blockstart + 2048 - 1;
			if ((blockstart > stop) || (blockstop < start))
			{
				skipsegment(p2fp);
				blockstart += 2048;
			}
			else
			{
				if (readsegment(p2fp) == -1)
					break;

				blockstop = blockstart + 1005 - 1;
				startpad = endpad = 0;
				samples = p2fp->p2seg.samples1;

				if ((blockstart <= stop) && (blockstop >= start))
				{
					if (start > blockstart) startpad = start - blockstart;
					if (stop < blockstop) endpad = blockstop - stop;
					memcpy(buf + curpos, samples + startpad, (blockstop - blockstart - startpad - endpad + 1) * 2);
					sexconv16_array(buf + curpos, blockstop - blockstart - startpad - endpad + 1);
					curpos += blockstop - blockstart - startpad - endpad + 1;
				}

				blockstart += 1005;
				blockstop = blockstart + 1021 - 1;
				startpad = endpad = 0;
				samples = p2fp->p2seg.samples2;

				if ((blockstart <= stop) && (blockstop >= start))
				{
					if (start > blockstart) startpad = start - blockstart;
					if (stop < blockstop) endpad = blockstop - stop;
					memcpy(buf + curpos, samples + startpad, (blockstop - blockstart - startpad - endpad + 1) * 2);
					sexconv16_array(buf + curpos, blockstop - blockstart - startpad - endpad + 1);
					curpos += blockstop - blockstart - startpad - endpad + 1;
				}

				blockstart += 1021;
				blockstop = blockstart + 22 - 1;
				startpad = endpad = 0;
				samples = p2fp->p2seg.samples3;

				if ((blockstart <= stop) && (blockstop >= start))
				{
					if (start > blockstart) startpad = start - blockstart;
					if (stop < blockstop) endpad = blockstop - stop;
					memcpy(buf + curpos, samples + startpad, (blockstop - blockstart - startpad - endpad + 1) * 2);
					sexconv16_array(buf + curpos, blockstop - blockstart - startpad - endpad + 1);
					curpos += blockstop - blockstart - startpad - endpad + 1;
				}

				blockstart += 22;
			}
		}
		else break;											/* ERROR */
	}
	if (feof(fp) || ((entry == p2fp->lastentry) && (blockstart > stop)) || ((p2fp->p2seg.recordwords != 0) && (status == -1)))
	{
		fseek(fp, startpos, SEEK_SET);
		curentry = entry;
		status = 0;
		recordwords = p2fp->p2seg.recordwords;
	}

	/*
	** There was an error.  Go back to the starting position...
	*/
	if (status == -1)
	{
		fseek(fp, startpos, SEEK_SET);
		return -1;
	}

	if (entrysize_p != NULL) *entrysize_p = recordwords;
	if (numwritten_p != NULL) *numwritten_p = curpos;
	p2fp->currententry = curentry;
	return 0;
}


int pcmseq2_getinfo(P2FILE *p2fp, int entry, long *entrysize_p, int *samplerate_p, unsigned long long *datetime_p, int *nentries_p)
{
	long *sizecache = p2fp->entry_size_cache;
	int *srcache = p2fp->entry_sr_cache;
	unsigned long long *timecache = p2fp->entry_time_cache;

	if (entrysize_p != NULL) *entrysize_p = 0;
	if (samplerate_p != NULL) *samplerate_p = 0;
	if (datetime_p != NULL) *datetime_p = 0;
	if (nentries_p != NULL) *nentries_p = 0;

	if ((entry <= 0) || ((p2fp->lastentry != -1) && (entry > p2fp->lastentry)))
		return -1;

	/* If we don't know the size, we don't know anything... */
	if (sizecache[entry] == -1)
		return -1;

	/*
	** Return what we can
	*/
	if ((entrysize_p != NULL) && (sizecache[entry] != -1))
		*entrysize_p = sizecache[entry];
	if ((samplerate_p != NULL) && (srcache[entry] != -1))
		*samplerate_p = srcache[entry];
	if ((datetime_p != NULL) && (timecache[entry] != -1))
		*datetime_p = timecache[entry];
	if ((nentries_p != NULL) && (p2fp->lastentry != -1))
		*nentries_p = p2fp->lastentry;

	return 0;
}


int pcmseq2_seektoentry(P2FILE *p2fp, int entry)
{
	FILE *fp;
	int curentry;
	long *poscache;

	if ((entry <= 0) || ((p2fp->lastentry != -1) && (entry > p2fp->lastentry)))
		return -1;

	fp = p2fp->fp;
	poscache = p2fp->entry_pos_cache;

	if (poscache[entry] != -1)
	{
		(void)fseek(fp, poscache[entry], SEEK_SET);
		curentry = entry;
		p2fp->currententry = curentry;
		return 0;
	}

	return -1;
}

static void skipsegment(P2FILE *p2fp)
{
	int type = p2fp->type;
	(void)fseek(p2fp->fp, (type == 1) ? 4140 : 4134, SEEK_CUR);
}

static int readentryhdr(P2FILE *p2fp)
{
	char *buf = p2fp->hdrbuf;
	char *initkey;
	int ret, type = p2fp->type;

	/*
	** Read in the entry header
	*/
	ret = fread((type == 1) ? buf : buf + 2, (type == 1) ? 56 : 54, 1, p2fp->fp);
	if (ret != 1) return -1;

	/*
	** Validate the initkey
	*/
	initkey = buf + 4;
	if ((initkey[0] != ' ') || (initkey[1] != '2'))
		return -1;

	/*
	** Read in everything else, byte-sex converting as necessary.
	** NOTE: I'm not converting the datetime[] ...
	*/
	p2fp->p2hdr.controlword = sexconv16(*(short *)(&buf[2]));
	p2fp->p2hdr.initkey = buf + 4;
	p2fp->p2hdr.datetime[0] = *(int *)(&buf[32]);
	p2fp->p2hdr.datetime[1] = *(int *)(&buf[36]);
	p2fp->p2hdr.segmentsize = sexconv32(*(int *)(&buf[40]));
	p2fp->p2hdr.pcmstart = sexconv32(*(int *)(&buf[44]));
	p2fp->p2hdr.gain = sexconv32(*(int *)(&buf[48]));
	p2fp->p2hdr.samplerate = sexconv32(*(int *)(&buf[52]));
	return 0;
}

static int readsegment(P2FILE *p2fp)
{
	char *buf = p2fp->segbuf;
	char *initkey, *matchkey;
	int ret, type = p2fp->type, off;

	/*
	** Read in the entry header
	*/
	ret = fread((type == 1) ? buf : buf + 2, (type == 1) ? 4140 : 4134, 1, p2fp->fp);
	if (ret != 1) return -1;

	/*
	** Validate the matchkey
	*/
	initkey = p2fp->p2hdr.initkey;
	matchkey = buf + 4;
	if ((matchkey[0] != ' ') || (matchkey[1] != '3') || (memcmp(initkey+2, matchkey+2, 16)))
		return -1;

	/*
	** Read in everything else, byte-sex converting as necessary.
	*/
	p2fp->p2seg.controlword1 = sexconv16(*(short *)(&buf[2]));
	p2fp->p2seg.matchkey = buf + 4;
	p2fp->p2seg.recordwords = sexconv32(*(int *)(&buf[32]));
	p2fp->p2seg.samples1 = (short *)(buf + 36);
	off = (type == 1) ? 0 : -2;
	p2fp->p2seg.controlword2 = sexconv16(*(short *)(&buf[2048 + off]));
	p2fp->p2seg.samples2 = (short *)(buf + 2050 + off);
	off = (type == 1) ? 0 : -4;
	p2fp->p2seg.controlword3 = sexconv16(*(short *)(&buf[4094 + off]));
	p2fp->p2seg.samples3 = (short *)(buf + 4096 + off);

	return 0;
}

static int peekcontrolword(P2FILE *p2fp, unsigned short *controlword_p)
{
	static char buf[4];
	int ret, type = p2fp->type, off;
	int jj;

	buf[0] = buf[1] = buf[2] = buf[3] = '\0';
	*controlword_p = 65535;
	off = (type == 1) ? 0 : 2;
	jj = ftell(p2fp->fp);
	ret = fread(buf + off, 4 - off, 1, p2fp->fp);
	jj = ftell(p2fp->fp);
	if (ret == -1)
		return -1;
	*controlword_p = sexconv16(*(unsigned short *)(buf + 2));
	fseek(p2fp->fp, (type == 1) ? -4 : -2, SEEK_CUR);
	return 0;
}


/*
** Scan through the pcm_seq2 file and determine the file positions and sample sizes 
** of each entry in the file
**
** The algorithm:
**	1) go to the end of the last entry of the file
**  2) read the last segment header of this entry to get its size.
**  3) from the size, determine the onset of this entry, and go there
**  4) repeat from (2) until you have reached the first entry.
*/
static int pcmseq2_indexfile(P2FILE *p2fp)
{
	FILE *fp = p2fp->fp;
	long *poscache = p2fp->entry_pos_cache;
	long *sizecache = p2fp->entry_size_cache;
	unsigned long long *timecache = p2fp->entry_time_cache;
	int *srcache = p2fp->entry_sr_cache;

	int type, curentry, tmpentry, numentries;
	long tmppos, tmpsize;
	struct stat buf;
	unsigned char tmpchr, rawseghdr[38], *seghdr, rawenthdr[58], *enthdr;
	int segmentsamples, entryhdrsize, segmentsize, recordwords, entrysegments;
	long deltapos;
	int tmpsr;
	unsigned long long tmptime;

	/*
	** Figure out the type
	*/
	fseek(fp, 0l, SEEK_SET);
	fread(&tmpchr, 1, 1, fp);
	if (tmpchr == 0x36)
		type = 1;
	else if (tmpchr == 0x03)
		type = 2;
	else
		return -1;

	/*
	** Get some convenience variables
	*/
	segmentsamples = 2048;
	entryhdrsize = (type == 1) ? 56 : 54;
	segmentsize = (type == 1) ? 4140 : 4134;
	seghdr = rawseghdr;
	if (type == 1) seghdr += 2;
	enthdr = rawenthdr;
	if (type == 1) enthdr += 2;

	/*
	** Figure out the size of the file, and go to beginning
	** of the last segment of the last entry.
	*/
	fstat(fileno(fp), &buf);
	fseek(fp, buf.st_size - segmentsize, SEEK_SET);
	curentry = 1;

	/*
	** Loop until we reach the beginning
	*/
	for (;;)
	{
		/*
		** Read and validate the segment header, and extract the recordwords field.
		** Add this to the cache in reversed entry order (which we'll fix when we're
		** done - ie: when we know how many entries there are).
		*/
		if ((fread(rawseghdr, 36, 1, fp) != 1) ||
			((seghdr[0] != 0x01) || (seghdr[1] != 0x00) || (seghdr[2] != 0x20) || (seghdr[3] != 0x33)) ||
			((type == 1) && ((rawseghdr[0] != 0xfc) || (rawseghdr[1] != 0x07))))
		{
			if ((curentry == 1) && (scantosegmentstart(p2fp) == 0))
			{
				/*
				** A pcm_seq2 file may have been interrupted while being written, resulting in an
				** incomplete segment.  We deal with this by scanning backwards to the beginning of
				** the last segment, going back 1 byte, and scanning backwards from there.
				*/
				continue;
			}
			else
			{
				fprintf(stderr, "pcm_seq2 ERROR: can't read or validate last segment hdr, entry (last - %d)\n", curentry - 1);
				return -1;
			}
		}
		memcpy(&recordwords, &(seghdr[30]), 4);
		recordwords = sexconv32(recordwords);
		if ((recordwords == -1) || (recordwords == 0))
		{
			/*
			** Gather seems to occasionally put -1 in the recordwords field of the last entry of a file;
			** this is bad for us... Here's a hack to get around it: it scans backwards in the file to
			** get to the beginning of the entry.
			*/
			fprintf(stderr, "pcm_seq2 WARNING: entry (last - %d) has invalid recordwords field.  Scanning...\n", curentry - 1);
			fseek(fp, 0 - 36, SEEK_CUR);
			tmppos = ftell(fp);
			if (pcmseq2_scantoentrystart(fp, type, segmentsize, entryhdrsize) == -1)
			{
				fprintf(stderr, "pcm_seq2 ERROR: unable to seek to beginning of entry.  Aborting.\n");
				return -1;
			}
			entrysegments = (tmppos + segmentsize - ftell(fp) - entryhdrsize) / (segmentsize);
			recordwords = entrysegments * segmentsamples;
		}
		else
		{
			/*
			** Round up the recordwords to include the entirety of the last segment
			** In other words, round it up to a multiple of 2048 samples (segmentsamples).
			**
			** The start of the entry should be at:
			**    Start-Of-Entry = End-Of-Entry - (EntrySegments * SegmentSize) - (EntryHdrSize)
			** However,
			**    Current-Pos = End-Of-Entry - SegmentSize + 36
			** So,
			**    End-Of-Entry = Current-Pos + SegmentSize - 36
			** And therefore,
			**    Start-Of-Entry = Current-Pos + SegmentSize - 36 - (EntrySegments * SegmentSize) - (EntryHdrSize)
			**
			** Go to and validate the Start-Of-Entry position.  If correct, cache it and the size.
			*/
			entrysegments = ((recordwords + segmentsamples - 1) / segmentsamples);
			deltapos = segmentsize - 36 - (entrysegments * segmentsize) - entryhdrsize;
			fseek(fp, deltapos, SEEK_CUR);
		}

		if ((fread(rawenthdr, 56, 1, fp) != 1) ||
			((enthdr[0] != 0x03) || (enthdr[1] != 0x00) || (enthdr[2] != 0x20) || (enthdr[3] != 0x32)) ||
			((type == 1) && ((rawenthdr[0] != 0x36) || (rawenthdr[1] != 00))))
		{
			deltapos = (segmentsize * -1) - 56;
			fseek(fp, deltapos, SEEK_CUR);
			if ((fread(rawenthdr, 56, 1, fp) != 1) ||
				((enthdr[0] != 0x03) || (enthdr[1] != 0x00) || (enthdr[2] != 0x20) || (enthdr[3] != 0x32)) ||
				((type == 1) && ((rawenthdr[0] != 0x36) || (rawenthdr[1] != 00))))
			{
				fprintf(stderr, "pcm_seq2 ERROR: can't read or validate entry hdr, entry (last - %d) (%lx)\n", curentry - 1, ftell(fp));
				fprintf(stderr, "%02x %02x %02x %02x\n", enthdr[0], enthdr[1], enthdr[2], enthdr[3]);
				return -1;
			}
		}
		poscache[curentry] = ftell(fp) - 56;
		sizecache[curentry] = recordwords;
		srcache[curentry] = (enthdr[50]) | (enthdr[51] << 8) | (enthdr[52] << 16) | (enthdr[53] << 24);
		timecache[curentry] = ( ((unsigned long long) enthdr[30]) |
								((unsigned long long) enthdr[31] << 8) |
								((unsigned long long) enthdr[32] << 16) |
								((unsigned long long) enthdr[33] << 24) |
								((unsigned long long) enthdr[34] << 32) |
								((unsigned long long) enthdr[35] << 40) |
								((unsigned long long) enthdr[36] << 48) |
								((unsigned long long) enthdr[37] << 56) );

		/*
		** Are we finished?
		*/
		if (poscache[curentry] == 0)
			break;

		/*
		** If not, go to the beginning of the last segment of the previous entry
		** And remember to account for (Current-Pos == Start-Of-Entry + 56)
		*/
		curentry++;
		fseek(fp, 0 - segmentsize - 56, SEEK_CUR);
	}

	/*
	** Reverse the entries (put them into the correct order)
	*/
	numentries = curentry;
	tmpentry = 1;
	while (curentry > tmpentry)
	{
		tmppos = poscache[tmpentry];
		poscache[tmpentry] = poscache[curentry];
		poscache[curentry] = tmppos;

		tmpsize = sizecache[tmpentry];
		sizecache[tmpentry] = sizecache[curentry];
		sizecache[curentry] = tmpsize;

		tmpsr = srcache[tmpentry];
		srcache[tmpentry] = srcache[curentry];
		srcache[curentry] = tmpsr;

		tmptime = timecache[tmpentry];
		timecache[tmpentry] = timecache[curentry];
		timecache[curentry] = tmptime;

		curentry--;
		tmpentry++;
	}

	p2fp->type = type;
	p2fp->lastentry = numentries;
	return 0;
}

/*
** This routine must be called when the current pos is at the start of a segment (usually the last segment)
** of an entry.  It quickly scans backwards to the beginning of the entry
*/
static int pcmseq2_scantoentrystart(FILE *fp, int type, int segmentsize, int entryhdrsize)
{
	long prevpos;
	int step;
	unsigned char rawseghdr[38], *seghdr;

	seghdr = rawseghdr;
	if (type == 1) seghdr += 2;

	/*
	** Make sure that we are actually on a valid seghdr
	*/
	if ((fread(rawseghdr, 36, 1, fp) != 1) ||
		((seghdr[0] != 0x01) || (seghdr[1] != 0x00) || (seghdr[2] != 0x20) || (seghdr[3] != 0x33)) ||
		((type == 1) && ((rawseghdr[0] != 0xfc) || (rawseghdr[1] != 0x07))))
	{
		fprintf(stderr, "ERROR: pcmseq2_scantoentrystart called from invalid file position\n");
		return -1;
	}
	fseek(fp, 0 - 36, SEEK_CUR);

	/*
	** Go backwards by 256 segments at a time, until we are in the previous entry.
	** Adjust the step rate if we are near the beginning of the file.
	*/
	step = 256;
	for (;;)
	{
		prevpos = ftell(fp);
		while ((step > 0) && (step * segmentsize > prevpos)) step = step / 2;
		fseek(fp, -1 * step * segmentsize, SEEK_CUR);
		if ((fread(rawseghdr, 36, 1, fp) != 1) ||
			((seghdr[0] != 0x01) || (seghdr[1] != 0x00) || (seghdr[2] != 0x20) || (seghdr[3] != 0x33)) ||
			((type == 1) && ((rawseghdr[0] != 0xfc) || (rawseghdr[1] != 0x07))))
		{
			/*
			** If validation failed, we must have gone into the previous entry.  So, go back to a good
			** position, and use a smaller step.
			*/
			fseek(fp, prevpos, SEEK_SET);
			step = step / 2;
			if (step > 0) continue;
		}
		else
		{
			/*
			** If, after stepping, we are still in the current entry, then undo the reading of the header,
			** and take another step
			*/
			fseek(fp, 0 - 36, SEEK_CUR);
			fprintf(stderr, "Stepped %d\n", step);
		}

		/*
		** We are done if even a single step backwards takes us out of the entry
		*/
		if (step == 0)
			break;
	}

	if (step == 0)
		fseek(fp, -1 * entryhdrsize, SEEK_CUR);

	return 0;
}


static int scantosegmentstart(P2FILE *p2fp)
{
	FILE *fp = p2fp->fp;
	char *buf = p2fp->segbuf;
	char *initkey;
	int i, type = p2fp->type, segsize;
	long startpos;

	segsize = (type == 1) ? 4140 : 4134;
	startpos = ftell(fp);

	/*
	** Need to read an entry hdr (any entry hdr in the file; we assume that the init_key
	** is conserved (at least the beginning of it)).  Thus, we read the first entry hdr.
	*/
	fseek(fp, 0L, SEEK_SET);
	if ((startpos < segsize) || (readentryhdr(p2fp) == -1))
	{
		fseek(fp, startpos, SEEK_SET);
		return -1;
	}

	/*
	** Now read the last 'segsize' bytes, and try to find a segment header
	** in there
	*/
	fseek(fp, startpos - segsize, SEEK_SET);
	if (fread(buf, segsize, 1, fp) != 1)
	{
		fseek(fp, startpos, SEEK_SET);
		return -1;
	}
	initkey = p2fp->p2hdr.initkey;
	for (i=0; i < segsize; i++)
	{
		if (memcmp(initkey + 2, buf + i, 14) == 0)
		{
			fseek(fp, startpos - segsize + i - 2 - ((type == 1) ? (4) : (2)), SEEK_SET);
			fprintf(stderr, "WARNING: last segment interrupted (size %d instead of %d)\n", segsize - i + 2 + ((type == 1) ? (4) : (2)), segsize);
			return 0;
		}
	}
	fseek(fp, startpos, SEEK_SET);
	return -1;
}


// code from pcmseq2_write

#define putshort(fd, val) { short temp = sexconv16(val); rc |= (fwrite(&temp, 2, 1, (fd)) != 1); }
#define putlong(fd, val) { long temp = sexconv32(val); rc |= (fwrite(&temp, 4, 1, (fd)) != 1); }
#define putchars(fd, ptr, len) { rc |= (fwrite(ptr, len, 1, (fd)) != 1); }
#ifdef sexconv_needed
#define putdata(fd, data, nsamples) \
	{\
		int n;\
		for (n=0; n < (nsamples); n++)\
			putshort(fd, data[n]);\
	}
#else
#define putdata(fd, data, nsamples) { rc |= (fwrite(ptr, (nsamples)*2, 1, (fd)) != 1); }
#endif

static short zeropad[2048];
static int zeropadzeroed = 0;

int pcmseq2_write_hdr(PCMFILE *fp)
{
	int rc = 0;
	char basename[8];
	FILE *fd = fp->outfp;
	unsigned long long lltime;
	unsigned char jj;

	/*
	** This is for pcmseq2_write_data()
	*/
	if (zeropadzeroed == 0)
	{
		memset(zeropad, '\0', 2048 * sizeof(short));
		zeropadzeroed = 1;
	}

	/*
	** Prepare the PCM_SEQ2 init_key field
	*/
	strncpy(basename, fp->name, 7);
	basename[7] = '\0';
	strcpy(fp->pcmseq3_key, "                            ");			/* IMPORTANT: 28 spaces, then null */
	strcpy(fp->pcmseq3_key, " 2");
	strcat(fp->pcmseq3_key, basename);
	fp->pcmseq3_key[strlen(basename) + 2] = ' ';
	sprintf(fp->pcmseq3_key + 21, "%3d   1", fp->entry);

	/*
	** Write the entry header
	*/
#ifdef VMS
	putshort(fd, 2 + 28 + 8 + 4 + 4 + 4 + 4);							/* record size, in bytes */
#endif
	putshort(fd, 0x03);													/* control word */
	putchars(fd, fp->pcmseq3_key, 28);									/* init_key */

	if (fp->timestamp == 0)
	{
		struct timeval curtime;

		gettimeofday(&curtime, NULL);
		lltime = ((unsigned long long)(curtime.tv_sec) - 18000LL) * 10000000LL;
		lltime += (unsigned long long)(curtime.tv_usec) * 10LL;
		lltime += 0x007c95674beb4000LL;	/* 64-bit date time */
	}
	else
	{
		lltime = ((unsigned long long)(fp->timestamp) - 18000LL) * 10000000LL;
		lltime += fp->microtimestamp * 10LL;
		lltime += 0x007c95674beb4000LL;	/* 64-bit date time */
	}
	jj = lltime & 0xffLL; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff00LL) >> 8; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff0000LL) >> 16; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff000000LL) >> 24; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff00000000LL) >> 32; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff0000000000LL) >> 40; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff000000000000LL) >> 48; fwrite(&jj, 1, 1, fd);
	jj = (lltime & 0xff00000000000000LL) >> 56; fwrite(&jj, 1, 1, fd);

	putlong(fd, 2048);													/* segment_size */
	putlong(fd, 0x1);													/* pcm_start */
	putlong(fd, 0x20f01);												/* gain?? */
	putlong(fd, fp->samplerate);										/* samplerate */

	/*
	** Convert the init_key field into a match_key field
	*/
	fp->pcmseq3_key[1] = '3';											/* Turn init_key into a match_key */
	fp->pcmseq3_cursamp = 2048;
	fp->pcmseq3_entrysize = 0;

	if ((rc != 0) || feof(fd) || ferror(fd))
		return -1;
	return 0;
}

/*
** This routine MUST get called with a block of 2048 samples!
*/
int pcmseq2_write_2048(PCMFILE *fp, short *data, int lastsegment)
{
	int rc = 0;
	int recordwords, towritesamples;
	short *ptr = data;
	FILE *fd = fp->outfp;

	fp->pcmseq3_entrysize += 2048;
	recordwords = (lastsegment == 0) ? 0 : (fp->pcmseq3_entrysize);
	if ((lastsegment != 0) && (data == NULL)) data = zeropad;

	/* First seg */
	ptr = data;
	towritesamples = 1005;
#ifdef VMS
	putshort(fd, 0x7fc);												/* record size (1 short) */
#endif
	putshort(fd, 0x01);													/* signifies first segment (1 short) */
	putchars(fd, fp->pcmseq3_key, 28);									/* match_key (28 bytes) */
	putlong(fd, recordwords);											/* total record words (1 long) */
	putdata(fd, ptr, towritesamples);									/* the first 1005 samples */
	ptr += towritesamples;

	/* Second seg */
	towritesamples = 1021;
#ifdef VMS
	putshort(fd, 0x7fc);												/* record size (1 short) */
#endif
	putshort(fd, 0x00);													/* signifies middle segment (1 short) */
	putdata(fd, ptr, towritesamples);									/* the next 1021 samples */
	ptr += towritesamples;

	/* Third, and last seg */
	towritesamples = 22;
#ifdef VMS
	putshort(fd, 0x2e);													/* record size (1 short) */
#endif
	putshort(fd, 0x02);													/* signifies last segment (1 short) */
	putdata(fd, ptr, towritesamples);									/* the last 22 samples */

	if ((rc != 0) || feof(fd) || ferror(fd))
		return -1;
	return 0;
}


/*
** This routine can be called with variable numbers of samples.
** To do this, we maintain our state between calls.
** The state variable consists of 'cursamp', which is the
** current sample from 0 to 2047.  It's initial value must be 2048.
**
** After calling pcmseq2_write_hdr(), this routine can be called 
** several times in a row with variable numbers of samples.  To
** complete the current entry, set the lastsegment parameter to 1
** on the last call.  If this is not desirable, this routine can
** also be called with nsamples == 0 and lastsegment == 1 to close
** the entry.
**
** The entry may be padded with zero samples at the end to reach
** a 2048-tuple sample count.  The correct size will be written
** in the last record's recordwords field.
*/
#define FIRSTSEG 1
#define MIDDLESEG 2
#define LASTSEG 3

int pcmseq2_write_data(PCMFILE *fp, short *data, int nsamples, int lastsegment)
{
	int rc = 0;
	long recordwords, cursamp, togo;
	short state, *ptr = data;
	FILE *fd = fp->outfp;

	fp->pcmseq3_entrysize += nsamples;
	recordwords = 0;

	cursamp = fp->pcmseq3_cursamp;
	if (cursamp < 1005) state = FIRSTSEG;
	else if (cursamp >= 2026) state = LASTSEG;
	else state = MIDDLESEG;

	if ((lastsegment != 0) && (nsamples == 0))
	{
		nsamples = 2048 - cursamp;
		ptr = zeropad;
	}
	while (nsamples > 0)
	{
		switch(state)
		{
			case FIRSTSEG:
				togo = 1005 - cursamp;									/* how many 'til we can finish FIRSTSEG */
				if (nsamples >= togo)									/* we have enough to finish FIRSTSEG */
				{
					putdata(fd, ptr, togo);								/* write 'togo' samples */
					ptr += togo;
					nsamples -= togo;
					cursamp = 1005;
					state = MIDDLESEG;									/* transition to MIDDLESEG state */
#ifdef VMS
					putshort(fd, 0x7fc);								/* write middleseg header */
#endif
					putshort(fd, 0x00);
				}
				else
				{
					putdata(fd, ptr, nsamples);							/* can't end state, so write what we can */
					ptr += nsamples;
					cursamp += nsamples;
					nsamples = 0;
				}
				break;
			case MIDDLESEG:
				togo = 2026 - cursamp;									/* how many 'til we can finish MIDDLESEG */
				if (nsamples >= togo)									/* we have enough to finish MIDDLESEG */
				{
					putdata(fd, ptr, togo);
					ptr += togo;
					nsamples -= togo;
					cursamp = 2026;
					state = LASTSEG;									/* transition to LASTSEG state */
#ifdef VMS
					putshort(fd, 0x2e);									/* write lastseg header */
#endif
					putshort(fd, 0x02);
				}
				else
				{
					putdata(fd, ptr, nsamples);							/* can't end state, so write what we can */
					ptr += nsamples;
					cursamp += nsamples;
					nsamples = 0;
				}
				break;
			case LASTSEG:
				togo = 2048 - cursamp;									/* how many 'til we can finish LASTSEG */
				if (nsamples >= togo)									/* we have enough to finish LASTSEG */
				{
					if (togo > 0)										/* initially, it will be 0 */
					{
						putdata(fd, ptr, togo);
						ptr += togo;
						nsamples -= togo;
					}
					cursamp = 0;
					state = FIRSTSEG;									/* transition to FIRSTSEG state */
					if ((lastsegment == 0) || (nsamples > 0))			/* if there is going to be more data... */
					{
						if ((lastsegment != 0) && (nsamples <= 2048))	/* we are almost finished */
							recordwords = fp->pcmseq3_entrysize;
#ifdef VMS
						putshort(fd, 0x7fc);							/* write firstseg header */
#endif
						putshort(fd, 0x01);
						putchars(fd, fp->pcmseq3_key, 28);
						fp->pcmseq3_poscache = ftell(fd);
						putlong(fd, recordwords);
					}
				}
				else
				{
					putdata(fd, ptr, nsamples);							/* can't end state, so write what we can */
					ptr += nsamples;
					cursamp += nsamples;
					nsamples = 0;
				}
				break;
		}
		if ((lastsegment != 0) && (nsamples == 0) && (cursamp > 0))
		{
			nsamples = 2048 - cursamp;
			ptr = zeropad;
			continue;
		}
	}
	if (lastsegment != 0)
	{
		fseek(fd, fp->pcmseq3_poscache, 0);
		putlong(fd, fp->pcmseq3_entrysize);
		fseek(fd, 0, 2);
	}

	fp->pcmseq3_cursamp = cursamp;

	if ((rc != 0) || feof(fd) || ferror(fd))
		return -1;
	return 0;
}
#undef FIRSTSEG
#undef MIDDLESEG
#undef LASTSEG

