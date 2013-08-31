/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The interface for the Layer class 
 */

#pragma once

#include <minerva/matrix/interface/Matrix.h>
namespace minerva
{
namespace neuralnetwork
{

class Layer
{
    public:
        typedef minerva::matrix::Matrix Matrix;
        typedef std::vector<Matrix> MatrixList;
        
        typedef MatrixList::iterator       iterator;
        typedef MatrixList::const_iterator const_iterator;


    public:
        Layer(unsigned totalBlocks = 0, size_t blockInput = 0, size_t blockOutput = 0)
        {
            m_sparseMatrix.resize(totalBlocks);
            for (auto i = m_sparseMatrix.begin(); i != m_sparseMatrix.end(); ++i)
            {
                //note that input is actually columns, and output is rows (since we take transpose of blocks to multiply)
                (*i).resize(blockInput, blockOutput);
            }
        }

        void initializeRandomly();
        Matrix runInputs(const Matrix& m) const;
        Matrix runReverse(const Matrix& m) const;
 
 	public:
 		void transpose();
    
    public:
        unsigned getInputCount()  const;
        unsigned getOutputCount() const;

    public:
        iterator       begin();
        const_iterator begin() const;

        iterator       end();
        const_iterator end() const;

    public:
              Matrix& operator[](size_t index);
        const Matrix& operator[](size_t index) const;
    
    public:
              Matrix& back();
        const Matrix& back() const;
    
    public:
    	void push_back(const Matrix& );
    
    public:
        unsigned int size() const;
        bool         empty() const; 
       
    private:
        std::vector<Matrix> m_sparseMatrix;

};

}
}
