/*****************************************************************************

Perceptual Evaluation of Speech Quality (PESQ)
ITU-T Recommendations P.862, P.862.1, P.862.2.
Version 2.0 - October 2005.

              ****************************************
              PESQ Intellectual Property Rights Notice
              ****************************************

DEFINITIONS:
------------
For the purposes of this Intellectual Property Rights Notice
the terms �Perceptual Evaluation of Speech Quality Algorithm�
and �PESQ Algorithm� refer to the objective speech quality
measurement algorithm defined in ITU-T Recommendation P.862;
the term �PESQ Software� refers to the C-code component of P.862.
These definitions also apply to those parts of ITU-T Recommendation 
P.862.2 and its associated source code that are common with P.862.

NOTICE:
-------
All copyright, trade marks, trade names, patents, know-how and
all or any other intellectual rights subsisting in or used in
connection with including all algorithms, documents and manuals
relating to the PESQ Algorithm and or PESQ Software are and remain
the sole property in law, ownership, regulations, treaties and
patent rights of the Owners identified below. The user may not
dispute or question the ownership of the PESQ Algorithm and
or PESQ Software.

OWNERS ARE:
-----------

1.	British Telecommunications plc (BT), all rights assigned
      to Psytechnics Limited
2.	Royal KPN NV, all rights assigned to OPTICOM GmbH

RESTRICTIONS:
-------------

The user cannot:

1.	alter, duplicate, modify, adapt, or translate in whole or in
      part any aspect of the PESQ Algorithm and or PESQ Software
2.	sell, hire, loan, distribute, dispose or put to any commercial
      use other than those permitted below in whole or in part any
      aspect of the PESQ Algorithm and or PESQ Software

PERMITTED USE:
--------------

The user may:

1.	Use the PESQ Software to:
      i)   understand the PESQ Algorithm; or
      ii)  evaluate the ability of the PESQ Algorithm to perform
           its intended function of predicting the speech quality
           of a system; or
      iii) evaluate the computational complexity of the PESQ Algorithm,
           with the limitation that none of said evaluations or its
           results shall be used for external commercial use.

2.	Use the PESQ Software to test if an implementation of the PESQ
      Algorithm conforms to ITU-T Recommendation P.862.

3.	With the prior written permission of both Psytechnics Limited
      and OPTICOM GmbH, use the PESQ Software in accordance with the
      above Restrictions to perform work that meets all of the following
      criteria:
      i)    the work must contribute directly to the maintenance of an
            existing ITU recommendation or the development of a new ITU
            recommendation under an approved ITU Study Item; and
      ii)   the work and its results must be fully described in a
            written contribution to the ITU that is presented at a formal
            ITU meeting within one year of the start of the work; and
      iii)  neither the work nor its results shall be put to any
            commercial use other than making said contribution to the ITU.
            Said permission will be provided on a case-by-case basis.


ANY OTHER USE OR APPLICATION OF THE PESQ SOFTWARE AND/OR THE PESQ
ALGORITHM WILL REQUIRE A PESQ LICENCE AGREEMENT, WHICH MAY BE OBTAINED
FROM EITHER OPTICOM GMBH OR PSYTECHNICS LIMITED. 

EACH COMPANY OFFERS OEM LICENSE AGREEMENTS, WHICH COMBINE OEM
IMPLEMENTATIONS OF THE PESQ ALGORITHM TOGETHER WITH A PESQ PATENT LICENSE
AGREEMENT. PESQ PATENT-ONLY LICENSE AGREEMENTS MAY BE OBTAINED FROM OPTICOM.


***********************************************************************
*  OPTICOM GmbH                    *  Psytechnics Limited             *
*  Naegelsbachstr. 38,             *  Fraser House, 23 Museum Street, *
*  D- 91052 Erlangen, Germany      *  Ipswich IP1 1HN, England        *
*  Phone: +49 (0) 9131 53020 0     *  Phone: +44 (0) 1473 261 800     *
*  Fax:   +49 (0) 9131 53020 20    *  Fax:   +44 (0) 1473 261 880     *
*  E-mail: info@opticom.de,        *  E-mail: info@psytechnics.com,   *
*  www.opticom.de                  *  www.psytechnics.com             *
***********************************************************************

Further information is also available from www.pesq.org

*****************************************************************************/

#include <stdio.h>
#include "pesq.h"
#include "dsp.h"

void make_stereo_file (char *stereo_path_name, SIGNAL_INFO *ref_info, SIGNAL_INFO *deg_info) {
    make_stereo_file2 (stereo_path_name, ref_info, deg_info-> data);
}

void make_stereo_file2 (char *stereo_path_name, SIGNAL_INFO *ref_info, float *deg) {

    long            i;
    long            h;
    short          *buffer;
    FILE           *outputFile;
    long            n;

    n = ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000) - 2 * SEARCHBUFFER * Downsample;     

    buffer = (short *) safe_malloc (2 * n * sizeof (short));
    
    if ((outputFile = fopen (stereo_path_name, "wb")) == NULL) {
        printf ("MakeStereoFile : cannot open output file %s!", stereo_path_name);
        return;
    }

    for (i = 0; i < n; i++) {
        h = (int) ref_info-> data [SEARCHBUFFER * Downsample + i] / 2;
        if (h < -32767) h = -32767;
        if (h > 32767)  h = 32767;
        h = (short) h;
        buffer [2*i] = (short) h;    
        h = (int) deg [SEARCHBUFFER * Downsample + i] / 2;
        if (h < -32767) h = -32767;
        if (h > 32767)  h = 32767;
        h = (short) h;
        buffer [2*i + 1] = (short) h;                
    }

    fwrite (buffer, sizeof (short) * 2, n, outputFile);
    
    fclose (outputFile);
    safe_free (buffer);
}

extern float InIIR_Hsos_16k [];
extern float InIIR_Hsos_8k [];
extern long InIIR_Nsos;

void select_rate( long sample_rate, long * Error_Flag, char ** Error_Type )
{
    if( Fs == sample_rate )
        return;
    if( Fs_16k == sample_rate )
    {
        Fs = Fs_16k;
        Downsample = Downsample_16k;
        InIIR_Hsos = InIIR_Hsos_16k;
        InIIR_Nsos = InIIR_Nsos_16k;
        Align_Nfft = Align_Nfft_16k;
        return;
    }
    if( Fs_8k == sample_rate )
    {
        Fs = Fs_8k;
        Downsample = Downsample_8k;
        InIIR_Hsos = InIIR_Hsos_8k;
        InIIR_Nsos = InIIR_Nsos_8k;
        Align_Nfft = Align_Nfft_8k;
        return;
    }

    (*Error_Flag) = -1;
    (*Error_Type) = "Invalid sample rate specified";    
}

int file_exist( char * fname )
{
    FILE * fp = fopen( fname, "rb" );
    if( fp == NULL )
        return 0;
    else
    {
        fclose( fp );
        return 1;
    }
}

void load_src( long * Error_Flag, char ** Error_Type,
         SIGNAL_INFO * sinfo, short* data, long n_samples, long fs)
{
    long Nsamples;
    long to_read;
    long read_count;
    long count;

    char s;
    float *read_ptr;
    short *p_input;

    Fs = fs;
    Nsamples = n_samples;
    sinfo-> Nsamples = Nsamples + 2 * SEARCHBUFFER * Downsample;

    sinfo-> data =
        (float *) safe_malloc( (sinfo-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof(float) );
    if( sinfo-> data == NULL )
    {
        *Error_Flag = 1;
        *Error_Type = "Failed to allocate memory for source file";
        printf ("%s!\n", *Error_Type);
        return;
    }
    read_ptr = sinfo->data;
    p_input = data;
    

    // padding
    for( read_count = SEARCHBUFFER*Downsample; read_count > 0; read_count-- )
      *(read_ptr++) = 0.0f;

    int i = Nsamples;
    while(i--){
        *(read_ptr++) = (float)(*(p_input++));
    }

    for( read_count = DATAPADDING_MSECS  * (Fs / 1000) + SEARCHBUFFER * Downsample;
         read_count > 0; read_count-- )
      *(read_ptr++) = 0.0f;

    sinfo-> VAD = safe_malloc( sinfo-> Nsamples * sizeof(float) / Downsample );
    sinfo-> logVAD = safe_malloc( sinfo-> Nsamples * sizeof(float) / Downsample );
    if( (sinfo-> VAD == NULL) || (sinfo-> logVAD == NULL))
    {
        *Error_Flag = 1;
        *Error_Type = "Failed to allocate memory for VAD";
        printf ("%s!\n", *Error_Type);
        return;
    }
}

void alloc_other( SIGNAL_INFO * ref_info, SIGNAL_INFO * deg_info, 
        long * Error_Flag, char ** Error_Type, float ** ftmp)
{
    *ftmp = (float *)safe_malloc(
       max( max(
            (*ref_info).Nsamples + DATAPADDING_MSECS  * (Fs / 1000),
            (*deg_info).Nsamples + DATAPADDING_MSECS  * (Fs / 1000) ),
           12 * Align_Nfft) * sizeof(float) );
    if( (*ftmp) == NULL )
    {
        *Error_Flag = 2;
        *Error_Type = "Failed to allocate memory for temporary storage.";
        printf ("%s!\n", *Error_Type);
        return;
    }
}

/* END OF FILE */
