#ifndef MODEL_HPP
#define MODEL_HPP

namespace hmlp
{
namespace model
{

template<typename FUNC, typename T>
class Classification : public VirtualModel<T>
{
  public:

    Classification()
    {
    };

   void Fit()
   {
   };

   std::vector<T> Predict( std::vector<T> & x )
   {
     std::vector<T> predict;

     return predict;
   };

  private:
};


template<typename FUNC, typename PARAM, typename DATA, typename T>
class Regression : public VirtualModel<T>
{
  public:

    Regression( FUNC *objective, PARAM *param, DATA *data )
    {
    };

    void Fit()
    {
    };

    std::vector<T> Predict( std::vector<T> & x )
    {
      std::vector<T> predict;

      return predict;
    };

  private:

    /** regularization */

    /** implement VirtualFunction */
    FUNC *objective;

    /** parameters for the objective function */
    PARAM *param;

    DATA *data;

}; /** end class Regression() */


template<typename T>
class VirtualModel
{
  public:

    VirtualModel() {};

    virtual void Fit() = 0;

    virtual std::vector<T> Predict( std::vector<T> & x ) = 0;

  private:

};

}; /** end namespace model */
}; /** end namespace hmlp */


#endif /** define MODEL_HPP */
