CXX      = g++
CXXFLAGS = -O2 -fopenmp -std=c++17
TARGET   = kmeans

$(TARGET): kmeans.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) kmeans.cpp

clean:
	rm -f $(TARGET) kmeans_results.csv kmeans_3d.png kmeans_3d.vtp
