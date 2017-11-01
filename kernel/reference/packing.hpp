#ifndef PACKING_HPP
#define PACKING_HPP

namespace hmlp
{


/** arbitrary packing routine */
template<size_t NB, typename T, typename TPACK>
struct pack_pbxib
{
  /** structure closure, e.g. ldx */


  /** 
   *  Loop over ib * pb of X to fill packX. Type cast from T to TPACK
   *  if necessary.
   */
  inline virtual void operator ()
  (
    /** k is the number cols, pc is the col offset, pb is the packed size */
    size_t k, size_t pc, size_t pb,
    /** m is the number rows, ic is the row offset, ib is the packed size */
    size_t m, size_t ic, size_t ib,
    /** input data in type T */
    T *X,
    /** packed data in type TPACK */
    TPACK *packX
  ) = 0;

}; /** end struct pack_pbxib */


/** arbitrary unpacking routine */
template<size_t NB, typename T, typename TPACK>
struct unpack_ibxjb
{
  /** structure closure, e.g. ldx, rs_c and cs_c */
  size_t rs_c = 0;
  size_t cs_c = 0;

  inline virtual void operator ()
  (
    size_t m, size_t ic, size_t ib,
    size_t n, size_t jc, size_t jb,
    T *X,
    TPACK *packX
  ) = 0;

}; /** end struct unpack_ibxjb */













/** column-major matrix packing routine */
template<int NB, typename T, typename TPACK>
struct pack2D_pbxib : public pack_pbxib<NB, T, TPACK>
{
  /** structure closure, e.g. ldx */
  bool trans = false;
  size_t ldx = 0;

  /** 
   *  Loop over ib * pb of X to fill packX. Type cast from T to TPACK
   *  if necessary.
   */
  inline virtual void operator ()
  (
    /** k is the number cols, pc is the col offset, pb is the packed size */
    size_t k, size_t pc, size_t pb,
    /** m is the number rows, ic is the row offset, ib is the packed size */
    size_t m, size_t ic, size_t ib,
    /** input data in type T */
    T *X,
    /** packed data in type TPACK */
    TPACK *packX
  )
  {
    T *x_pntr[ NB ];

    if ( trans )
    {
      /** ( pc, ic ) offset  */
      X += ( ic * ldx + pc );

      for ( auto i = 0; i < ib; i ++ )
      {
        x_pntr[ i ] = X + ldx * i;
      }
      for ( auto i = ib; i < NB; i ++ )
      {
        x_pntr[ i ] = X;
      }

      for ( auto p = 0; p < pb; p ++ ) 
      {
        for ( auto i = 0; i < ib; i ++ )
        {
          *packX ++ = *x_pntr[ i ] ++;
        }
        for ( auto i = ib; i < NB; i ++ )
        {
          *packX ++ = 0;
        }
      }
    }
    else 
    {
      /** ( ic, pc ) offset  */
      X += ( pc * ldx + ic );

      for ( auto i = 0; i < ib; i ++ )
      {
        x_pntr[ i ] = X + i;
      }
      for ( auto i = ib; i < NB; i ++ )
      {
        x_pntr[ i ] = X;
      }

      for ( auto p = 0; p < pb; p ++ )
      {
        for ( auto i = 0; i < ib; i ++ )
        {
          *packX = *x_pntr[ i ];
          packX ++;
          x_pntr[ i ] += ldx;
        }
        for ( auto i = ib; i < NB; i ++ )
        {
          *packX ++ = 0;
        }
      }
    }
  };
}; /** end struct pack2D_pbxib */





/** arbitrary unpacking routine */
template<size_t NB, typename T, typename TPACK>
struct unpack2D_ibxjb : public unpack_ibxjb<NB, T, TPACK>
{
  /** structure closure, e.g. ldx */

  inline virtual void operator ()
  (
    size_t m, size_t ic, size_t ib,
    size_t n, size_t jc, size_t jb,
    T *X,
    TPACK *packX
  )
  {
    for ( size_t j = 0; j < jb; j ++ )
      for ( size_t i = 0; i < ib; i ++ )
        X[ ( jc + j ) * this->cs_c + ( ic + i ) * this->rs_c ] 
          = packX[ j * NB + i ];
  };

}; /** end struct unpack2D_ibxjb */




}; /** end namespace hmlp */


#endif /** define PACKING_HPP */
