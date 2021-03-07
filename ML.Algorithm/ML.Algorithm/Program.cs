using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Algorithm
{
    class Program
    {
        static void Main(string[] args)
        {
            KNNClassifier classifier = new KNNClassifier();
            classifier.Train(GenerateMovieData(200));
            int k = 5;
            object result = classifier.Classify(new double[] { 45, 0}, k);
            Console.WriteLine(result);
            Console.ReadKey();
        }

        static IEnumerable<TrainingModel> GenerateMovieData(int count)
        {
            Random random = new Random();
            var data = new List<TrainingModel>();
            for(int i = 0; i < count; i++)
            {
                int age = random.Next(5, 60);
                int gender = random.Next(0, 1);
                data.Add(new TrainingModel()
                {
                    Values = new double[] { age, gender },
                    Class = i % 2 == 0 ? "Likes Movies" : "Doesn't Like Movies"
                });
            }
            return data;
        }
    }

    public class TrainingModel : ITrainingModel
    {
        public double[] Values { get; set; }
        public object Class { get; set; }
    }

    public interface ITrainingModel
    {
        double[] Values { get; set; }
        object Class { get; set; }
    }

    public interface IClassifier
    {
        void Train(IEnumerable<ITrainingModel> trainingData);
        void TrainMore(IEnumerable<ITrainingModel> additionalTrainingData);
        object Classify(double[] value, int nearestNeighbours);
    }

    public class KNNClassifier : IClassifier
    {
        private IEnumerable<ITrainingModel> _trainingData;

        /// <summary>
        /// Trains the algorithm with new data
        /// </summary>
        /// <param name="trainingData"></param>
        public void Train(IEnumerable<ITrainingModel> trainingData)
        {
            _trainingData = trainingData;
        }

        /// <summary>
        /// Trains the algorithm with addittional data.
        /// </summary>
        /// <param name="additionalTrainingData"></param>
        public void TrainMore(IEnumerable<ITrainingModel> additionalTrainingData)
        {
            if (_trainingData == null)
            {
                _trainingData = additionalTrainingData;
                return;
            }

            var currentData = _trainingData.ToList();
            currentData.AddRange(additionalTrainingData);
            _trainingData = currentData;
        }

        /// <summary>
        /// Classifies the provided data.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="nearestNeighbours"></param>
        /// <returns></returns>
        public object Classify(double[] values, int nearestNeighbours)
        {
            var distanceData = new Dictionary<ITrainingModel, double>();

            foreach (var model in _trainingData)
            {
                distanceData.Add(model, CalculateEuclideanDistance(model.Values, values));
            }

            return distanceData.OrderBy(x => x.Value)
                .Take(nearestNeighbours)
                .GroupBy(x => x.Key.Class)
                .OrderByDescending(group => group.Count())
                .Select(group => group.Key).First();
        }

        private double CalculateEuclideanDistance(double[] source, double[] toCompare)
        {
            if (source.Length != toCompare.Length)
                throw new InvalidDataException("Data format is invalid");

            double distance = 0;

            for (int index = 0; index < source.Length; index++)
            {
                distance += Math.Pow(toCompare[index] - source[index], 2);
            }

            return Math.Sqrt(distance);
        }
    }

    public class InvalidDataException : Exception
    {
        public InvalidDataException(string message) : base(message) { }
    }
}
