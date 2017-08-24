#ifndef ROOT_HPP
#define ROOT_HPP

#include <assert.h>
#include <typeinfo>
#include <algorithm>
#include <random>
#include <limits>
#include <cstddef>
#include <math.h>

namespace hmlp
{
namespace root
{

template<typename T>
class RootFinderBase
{
  public:

		RootFinderBase() {};

		virtual std::pair<T, T> Initialize() = 0;

		virtual std::pair<T, T> Iterate() = 0;

		virtual std::pair<T, T> Solve( T user_x_lower, T user_x_upper )
		{
			/** setup user-specific search region */
			x_lower = user_x_lower;
			x_upper = user_x_upper;
			x = ( x_lower + x_upper ) / 2.0;

			auto root = Initialize();
			auto previous = root;

      size_t iter = 0;
			do
			{
        auto previous = root; 
        root = Iterate(); 
				iter ++;
        if ( !( iter % 50 ) ) printf( "iter %4lu f(x) %E\n", iter, root.second );
			} while ( !Terminate( iter, root, previous ) );

			return root;
		};

		T x = 0.0;

		T x_lower = 0.0;

		T x_upper = 0.0;

		void SetupTerminationCriteria( size_t niter, T tol )
		{
		  this->use_defined_termination_criteria = true;
			this->niter = niter;
			this->tol = tol;
		};

		bool ReachMaximumIteration( size_t iter ) 
		{
			return ( iter < niter );
		};

		bool Terminate( size_t iter, std::pair<T, T> &now, std::pair<T, T> &previous )
		{
			bool doterminate = false;
			if ( std::fabs( now.second ) < std::numeric_limits<T>::epsilon() )
			{
				return true;
			}

	    if ( use_defined_termination_criteria )
			{
				if ( iter >= niter ) doterminate = true;
        if ( std::fabs( ( now.second - previous.second ) / previous.second ) < tol )
					doterminate = true;
			}
			return doterminate;
		};

	private:

		bool use_defined_termination_criteria = false;

		/** */
		size_t niter = 10;

		T tol = 1E-5;
};


/**
 *  @brief This is not thread safe
 */ 
template<typename FUNC, typename T>
class Bisection : public RootFinderBase<T>
{
	public:

		Bisection( FUNC *func ) 
		{
			this->func = func;
		};

		std::pair<T, T> Initialize()
    {
			T &x_lower = this->x_lower;
			T &x_upper = this->x_upper;

			/** evaluate two end points */
			f_lower = func->F( x_lower );
			f_upper = func->F( x_upper );

			/** must be in different signs */
			if ( ( f_lower < 0.0 && f_upper < 0.0 ) || ( f_lower > 0.0 && f_upper > 0.0 ) )
			{
				printf( "endpoints do not straddle y = 0\n" );
				exit( 1 );
			}

			/** initialize the guess */
			std::pair<T, T> root( ( x_lower + x_upper ) / 2.0, ( f_lower + f_upper ) / 2.0 );

			return root;
		};


		std::pair<T, T> Iterate()
		{
			T x_bisect, f_bisect;

			T &x_lower = this->x_lower;
			T &x_upper = this->x_upper;

			if ( f_lower == 0.0 ) return std::pair<T, T>( x_lower, f_lower );
			if ( f_upper == 0.0 ) return std::pair<T, T>( x_upper, f_upper );

			/** bisection */
			x_bisect = ( x_lower + x_upper ) / 2.0;
			f_bisect = func->F( x_bisect );
			if ( f_bisect == 0.0 ) 
			{
				x_lower = x_bisect;
				x_upper = x_bisect;
			  return std::pair<T, T>( x_bisect, f_bisect );
			}

			/** discard the half of the interval which doesn't contain the root. */
			if ( ( f_lower > 0.0 && f_bisect < 0.0 ) || ( f_lower < 0.0 && f_bisect > 0.0 ) )
			{
				x_upper = x_bisect;
				f_upper = f_bisect;
			  return std::pair<T, T>( 0.5 * ( x_lower + x_upper ), f_upper );
			}
			else
			{
				x_lower = x_bisect;
				f_lower = f_bisect;
			  return std::pair<T, T>( 0.5 * ( x_lower + x_upper ), f_lower );
			}
		}; /** end Iterate() */

	private:

		FUNC *func = NULL;

		/** function value lower bound */
		T f_lower = 0.0;

		/** function value upper bound */
		T f_upper = 0.0;

}; /** end class Bisection */
	

template<typename FUNC, typename T>
class Newton : public RootFinderBase<T>
{
	public:

		Newton( FUNC *func ) 
		{
			this->func = func;
		};

		std::pair<T, T> Initialize()
		{
			T x_bisect, f_bisect, df_bisec;

			T &x_lower = this->x_lower;
			T &x_upper = this->x_upper;

			if ( !func->HasdF() )
			{
				printf( "Newton(): no first order derivity provided\n" );
				exit( 1 );
			};

			x_bisect = ( x_lower + x_upper ) / 2.0;
			f = func->F( x_bisect );
			df = func->dF( x_bisect );

			/** initialize the guess */
			std::pair<T, T> root( x_bisect, f );

			return root;
		};

		std::pair<T, T> Iterate()
		{
      auto &x = this->x;

      if ( df == 0.0 ) 
			{
			  printf( "Newton(): derivative is zero\n" );
				return std::pair<T, T>( x, f );
			}

			/** gradient decent */
			x -= ( f / df );

			if ( func->HasFdF() )
			{
			  auto fdf = func->FdF( x );
				f = fdf.first;
				df = fdf.second;
			}
			else
			{
				f = func->F( x );
				df = func->dF( x );
			}

			if ( f != f ) 
			{
				printf( "Newton(): f( x ) is infinite\n" );
			}

			if ( df != df ) 
			{
				printf( "Newton(): f'( x ) is infinite\n" );
			}

			return std::pair<T, T>( x, f );
		};

	private:

		FUNC *func = NULL;

		T f = 0.0;

		T df = 0.0;

}; /** end class Newton */


}; /** end namespace root */
}; /** end namespace hmlp */

#endif /** define ROOT_HPP */
