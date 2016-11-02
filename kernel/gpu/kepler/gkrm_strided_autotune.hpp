#ifdef PRECISION_d
  switch ( shape )
  {
    case 0: // nn
      {
        if ( k < 32 ) {
          if ( k == 8 && n == 24 ) {
            gkrm_strided_macro(false,false,NN,32);
          }
          else if ( n < 32 ) {
            gkrm_strided_macro(false,false,NN,49);
          }
          else {
            gkrm_strided_macro(false,false,NN,158);
          }
        }
        else {
          if ( m < 80 ) {
            gkrm_strided_macro(false,false,NN,93);
          }
          else { 
            gkrm_strided_macro(false,false,NN,158);
          }
        }
      }
      break;
    case 1: // nt
      {
        if ( k < 128 ) { 
          gkrm_strided_macro(false,true,NT,160);
        }
        else {
          if ( m < 256 ) 
            gkrm_strided_macro(false,true,NT,160);
          else 
            gkrm_strided_macro(false,true,NT,190);
        }
      }
      break;
    case 3: // tn
      {
        if ( k < 64 ) {
          gkrm_strided_macro(true,false,TN,207);
        }
        else {
          if ( m < 256 ) {
            gkrm_strided_macro(true,false,TN,207);
          }
          else {
            gkrm_strided_macro(true,false,TN,209);
          }
        }
      }
      break;
    case 4: // tt
      {
        if ( k < 128 ) {
          gkrm_strided_macro(true,true,TT,81);
        }
        else {
          if ( m < 256 ) {
            gkrm_strided_macro(true,true,TT,81);
          }
          else {
            gkrm_strided_macro(true,true,TT,85);
          }
        }
      }
      break;
    default:;
  }
#endif

#ifdef PRECISION_s
  switch(shape)
  {
    case 0: // nn
      {
        if (k < 64) {
          if (k == 8 && n == 24) {
            gkrm_strided_macro(false,false,NN,512);
          }
          else if (n < 32) {
            gkrm_strided_macro(false,false,NN,510);
          }
          else {
            gkrm_strided_macro(false,false,NN,504);
          }
        }
        else {
          gkrm_strided_macro(false,false,NN,518);
        }
      }
      break;
    case 1: // nt
      {
        gkrm_strided_macro(false,true,NT,734);
      }
      break;
    case 3: // tn
      {
        if (k < 64) {
          gkrm_strided_macro(true,false,TN,654);
        }
        else {
          gkrm_strided_macro(true,false,TN,666);
        }
      }
      break;
    case 4: // tt
      {
        if (k < 128) {
            if (m < 128) {
              gkrm_strided_macro(true,true,TT,275);
            }
            else {
              gkrm_strided_macro(true,true,TT,312);
            }
        }
        else {
          gkrm_strided_macro(true,true,TT,312);
        }
      }
    break;
  }
#endif
