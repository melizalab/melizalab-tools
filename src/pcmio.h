/*
** pcmio.h
**
** header file for libdataio, a library for reading certain types of data
** such as sampled sound/electrical data, labels (named events occuring at
** specific times), or spike times of events for neuronal action potential
** firing.   This library is primarily used by my programs aplot and by the
** a-ware set of programs for scientific data collection and analysis.
**
** Copyright (C) by Amish S. Dave 2003
**
** This code is distributed under the GNU GENERAL PUBLIC LICENSE (GPL)
** Version 2 (June 1991). See the "COPYING" file distributed with this software
** for more info.
*/

#ifndef PCMIOHEADER
#define PCMIOHEADER

extern int pcm_errno;
extern int sys_nerr;

#ifdef linux
#ifdef MADV_SEQUENTIAL
#undef MADV_SEQUENTIAL
#endif
#define MADV_SEQUENTIAL
#define madvise(addr,len,flags)
#endif

#define PCMIOMMAP 1
#define PCMIOMALLOC 2
#define PCMIOINCENTRY 3
#define PCMIODECENTRY 4
#define PCMIOSETTIME 5
#define PCMIOSETSR 6
#define PCMIOGETSIZE 7		   /* Get the size in samples - shortcut to using pcm_stat() - arg = int* */
#define PCMIOGETSR 8		   /* Get the samplerate in Hz - shortcut to using pcm_stat() - arg = int* */
#define PCMIOGETENTRY 9		   /* Get the entry - shortcut to using pcm_stat() - arg = int* */
#define PCMIOGETTIME 10		   /* Get the timestamp - shortcut to using pcm_stat() - arg = (long *) */
#define PCMIOGETCAPS 11		   /* Get the capabilities of this file format... Includes MULTENTRY */
#define PCMIOSETTIMEFRACTION 12	
#define PCMIOGETTIMEFRACTION 13	   /* Get the timestamp's microseconds fraction, arg = (long *) */
#define PCMIOGETNENTRIES 14	   /* Get the number of entries in this file */

#define PCMIOCAP_MULTENTRY 1	   /* This file format can hold >1 entry... */
#define PCMIOCAP_SAMPRATE 2	   /* This file format stores samplerate info... */

typedef struct pcmfilestruct
{
	char *name;
	int flags;
	int entry;
	int fd;
	void *addr;
	int len;
	short *bufptr;
	int buflen;
	int memalloctype;
	int (*open)();
	void (*close)();
	int (*read)();
	int (*write)();
	int (*seek)();
	int (*ctl)();
	int (*stat)();
	char *tempnam;
	int samplerate;
	int timestamp;
	long microtimestamp;
	int nentries;
	void *p2file;				/* For PCMSEQ2 */
	int pcmseq3_entrysize;		/* For PCMSEQ2 */
	char pcmseq3_key[29];		/* For PCMSEQ2 */
	long pcmseq3_cursamp;		/* For PCMSEQ2 */
	long pcmseq3_poscache;		/* For PCMSEQ2 */
	int entrystarted;			/* For PCMSEQ2 */
	FILE *outfp;
	int wav_nbytes_addr1;
	int wav_nbytes_addr2;
} PCMFILE;

struct pcmstat
{
	int	entry;
	int	nsamples;
	int	samplerate;
	int timestamp;
	long microtimestamp;
	int capabilities;
	int nentries;
};


/* Entry header */
typedef struct
{
	unsigned short		recordsize;
	unsigned short		controlword;
	char 				*initkey;         /* 28 bytes */
	int					datetime[2];
	unsigned int		segmentsize;
	unsigned int		pcmstart;
	unsigned int		gain;
	unsigned int		samplerate;
} p2header;

/* Segment */
typedef struct
{
	/** First seg **/
	unsigned short		recordsize1;
	unsigned short		controlword1;
	char 				*matchkey;        /* 28 bytes */
	unsigned int		recordwords;
	short				*samples1;        /* 1005 samples */
	/** Second seg **/
	unsigned short		recordsize2;
	unsigned short		controlword2;
	short				*samples2;        /* 1021 samples */
	/** Third seg **/
	unsigned short		recordsize3;
	unsigned short		controlword3;
	short				*samples3;        /* 22 samples */
} p2segment;

#define CACHESIZE 500

typedef struct
{
	FILE *fp;

	int currententry;
	int lastentry;
	int type;
	long entry_pos_cache[CACHESIZE];
	long entry_size_cache[CACHESIZE];
	int entry_sr_cache[CACHESIZE];
	unsigned long long entry_time_cache[CACHESIZE];

	p2header p2hdr;
	p2segment p2seg;

	char hdrbuf[56];
	char segbuf[4140];
} P2FILE;


PCMFILE *pcm_open(char *, char *);
void pcm_close(PCMFILE *);
int pcm_read(PCMFILE *fp, short **buf_p, int *nsamples_p);
int pcm_seek(PCMFILE *fp, int entry);
int pcm_ctl(PCMFILE *fp, int request, void *arg);
int pcm_stat(PCMFILE *fp, struct pcmstat *buf);
int pcm_write(PCMFILE *fp, short *buf, int nsamples);

int pcmseq_recognizer(PCMFILE *);
int pcmseq_open(PCMFILE *);
void pcmseq_close(PCMFILE *fp);
int pcmseq_read(PCMFILE *fp, short **buf_p, int *nsamples_p);
int pcmseq_seek(PCMFILE *fp, int entry);
int pcmseq_ctl(PCMFILE *fp, int request, void *arg);
int pcmseq_stat(PCMFILE *fp, struct pcmstat *buf);
int pcmseq_write(PCMFILE *fp, short *buf, int nsamples);


P2FILE *pcmseq2_open(char *filename);
void pcmseq2_close(P2FILE *p2fp);
int pcmseq2_read(P2FILE *p2fp, int entry, long start, long stop, short *buf, long *numwritten_p, long *entrysize_p);
int pcmseq2_getinfo(P2FILE *p2fp, int entry, long *entrysize_p, int *samplerate_p, unsigned long long *datetime_p, int *nentries_p);
int pcmseq2_seektoentry(P2FILE *p2fp, int entry);
int pcmseq2_write_hdr(PCMFILE *fp);
int pcmseq2_write_2048(PCMFILE *fp, short *data, int lastsegment);
int pcmseq2_write_data(PCMFILE *fp, short *data, int nsamples, int lastsegment);

#endif

