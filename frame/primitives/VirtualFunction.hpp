#ifndef VIRTUALFUNCTION_HPP
#define VIRTUALFUNCTION_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <limits>
#include <cstddef>
#include <math.h>

namespace hmlp
{

template<typename T>
class VirtualFunction
{
	public:

		VirtualFunction() {};

		virtual T F( T x ) = 0;

		virtual T dF( T x ) = 0;

		/** return Fx and dFx */
		virtual std::pair<T, T> FdF( T x ) = 0;

	private:
		
}; /** end class VirtualFunction */


template<typename PARAM, typename T>
class FunctionBase : public VirtualFunction<T>
{
	public:

		FunctionBase() {};

		T F( T x )
		{
			assert( f );
			return (*f)( x, params );
		};

		T dF( T x )
		{
			assert( df );
			return (*df)( x, params );
		};

		std::pair<T, T> FdF( T x )
		{
			assert( fdf );
			T Fx = 0.0, dFx = 0.0;
      (*fdf)( x, params, &Fx, &dFx );
			return std::pair<T, T>( Fx, dFx );
		};

    void SetupParameters( PARAM *user_params )
    {
      this->params = user_params;
    };

		void SetupF( T (*f)( T x, PARAM *user_params ) )
		{
			this->f = f;
		};

		void SetupdF( T (*df)( T x, PARAM *user_params ) )
		{
			this->df = df;
			hasdf = true;
		};

		void SetupFdF( T (*fdf)( T x, PARAM *user_params, T *Fx, T *dFX ) )
		{
			this->fdf = fdf;
			hasfdf = true;
		};

		bool HasdF()
		{
			return hasdf;
		};

		bool HasFdF()
		{
			return hasfdf;
		};

	private:

		/** gsl compatable parameter strcuture */
		PARAM *params;

		/** gsl compatable function pointer */
		T (*f)( T x, PARAM *user_params ) = NULL;

		/** gsl compatable 1st order derivity */
		T (*df)( T x, PARAM *user_params ) = NULL;

		/** gsl compatable function pointer and 1st order derivity */
		void (*fdf)( T x, PARAM *user_params, T *Fx, T *dFx ) = NULL;

		bool hasdf = false;

		bool hasfdf = false;

}; /** end class FunctionBase */


}; /** end namespace hmlp */

#endif /** define VIRTUALFUNCTION_HPP */
