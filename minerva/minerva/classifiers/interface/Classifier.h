/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The header of the class to classify test images into gestures
 */

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/video/interface/ImageVector.h>

// Standard Library Includes
#include <vector>
#include <string>

namespace minerva
{

namespace classifiers
{

typedef std::vector<std::string> GestureVector;

class Classifier
{
    public:
    	typedef model::ClassificationModel   ClassificationModel;
    	typedef video::ImageVector           ImageVector;
        typedef neuralnetwork::NeuralNetwork NeuralNetwork;
        typedef matrix::Matrix               Matrix;

    public:
        Classifier(const ClassificationModel& model) : m_classificationModel(model)
        {

        }
    
        GestureVector classify(const ImageVector& images);

    	unsigned getInputFeatureCount();
    	
    private:
        void loadFeatureSelector();
        void loadClassifier();
        Matrix detectGestures(const ImageVector& images);
        GestureVector pickMostLikelyGesture(const Matrix& m);
    
    	
    private:
        ClassificationModel m_classificationModel;

        /* These are read from disk */
        NeuralNetwork m_featureSelectorNetwork;
        NeuralNetwork m_classifierNetwork;

        /* These are read from test images */
        ImageVector m_inputImages;

        /* Vector of values for all possible gestures */
        GestureVector m_gestures;
};

} //end classifiers
} //end minerva
