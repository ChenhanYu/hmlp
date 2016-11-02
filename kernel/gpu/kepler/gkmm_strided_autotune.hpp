#ifdef PRECISION_d
  switch ( shape )
  {
    case 0: // nn
      {
        if ( k < 32 ) {
          if ( k == 8 && n == 24 ) {
            gkmm_strided_macro(false,false,NN,32);
          }
          else if ( n < 32 ) {
            gkmm_strided_macro(false,false,NN,49);
          }
          else {
            gkmm_strided_macro(false,false,NN,158);
          }
        }
        else {
          if ( m < 80 ) {
            gkmm_strided_macro(false,false,NN,93);
          }
          else { 
            gkmm_strided_macro(false,false,NN,158);
          }
        }
      }
      break;
    case 1: // nt
      {
        if ( k < 128 ) { 
          gkmm_strided_macro(false,true,NT,160);
        }
        else {
          if ( m < 256 ) 
            gkmm_strided_macro(false,true,NT,160);
          else 
            gkmm_strided_macro(false,true,NT,190);
        }
      }
      break;
    case 3: // tn
      {
        if ( k < 64 ) {
          gkmm_strided_macro(true,false,TN,207);
        }
        else {
          if ( m < 256 ) {
            gkmm_strided_macro(true,false,TN,207);
          }
          else {
            gkmm_strided_macro(true,false,TN,209);
          }
        }
      }
      break;
    case 4: // tt
      {
        if ( k < 128 ) {
          gkmm_strided_macro(true,true,TT,81);
        }
        else {
          if ( m < 256 ) {
            gkmm_strided_macro(true,true,TT,81);
          }
          else {
            gkmm_strided_macro(true,true,TT,85);
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
            gkmm_strided_macro(false,false,NN,512);
          }
          else if (n < 32) {
            gkmm_strided_macro(false,false,NN,510);
          }
          else {
            gkmm_strided_macro(false,false,NN,504);
          }
        }
        else {
          gkmm_strided_macro(false,false,NN,518);
        }
      }
      break;
    case 1: // nt
      {
        gkmm_strided_macro(false,true,NT,734);
      }
      break;
    case 3: // tn
      {
        if (k < 64) {
          gkmm_strided_macro(true,false,TN,654);
        }
        else {
          gkmm_strided_macro(true,false,TN,666);
        }
      }
      break;
    case 4: // tt
      {
        if (k < 128) {
            if (m < 128) {
              gkmm_strided_macro(true,true,TT,275);
            }
            else {
              gkmm_strided_macro(true,true,TT,312);
            }
        }
        else {
          gkmm_strided_macro(true,true,TT,312);
        }
      }
      break;
    default:;
  }
#endif
