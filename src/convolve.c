/* @(#)convolve.c
 *
 * Functions for computing convolutions
 */
#include <math.h>

void
discreteconv(const double *times, int ntimes, const double *kern, int nkern,
             double kdt, double onset, double offset, double odt,
             double *out)
{
        int ipt;
        int NT = (int)ceil((offset - onset) / odt);
        double W_dur = nkern * kdt;

        for (ipt = 0; ipt < ntimes; ipt++) {
                double cur_point = times[ipt];

                // check that the time is within the analysis window
                if (cur_point < onset || cur_point > offset)
                        continue;

                // compute time relative to the initial point of the grid
                cur_point -= onset;

                // find the nearest grid point to the left of the current point
                int dt = (int)floor( cur_point / odt);

                // If kernel is translated to the time grid point, the relative
                // location of the current point in the support of the kernel
                double rel_loc = cur_point - odt * (double)dt;

                while (rel_loc <= W_dur && dt < NT) {
                    /* Use linear interpolation to compute the change
                       to each time grid point.

                       Among the grid points for function [f], find
                       the right most one to the left to the current
                       point.
                    */
                    double dR = rel_loc / kdt;
                    int  drel_loc = (int)floor(dR);
                    if (drel_loc < nkern ) {
                            double dx = dR - drel_loc;
                            out[dt] += (1.0 - dx) * kern[drel_loc];
                            if ( drel_loc < nkern - 1)
                                    out[dt] += dx * kern[drel_loc+1];
                    }
                    dt ++;
                    rel_loc += odt;
            }
    }
}
