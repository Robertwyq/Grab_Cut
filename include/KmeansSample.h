//
// Created by Robert on 2019/6/5.
//

#ifndef GRAB_CUT_KMEANSSAMPLE_H
#define GRAB_CUT_KMEANSSAMPLE_H
const  double eps=1e-3;
const  int Max_ITE =100;
class Kmeans{
public:
    Kmeans(int Dim_NUM = 1, int K = 1);
    ~Kmeans();

    void Init(double *data, int N);
    void Cluster(double *data, int N, int *label);

    double **means;
private:
    /**
     * Dimension: For RGB is 3
     */
    int dim;
    /**
     * Cluster: the number of labels
     */
    int k;
    /**
     * Find the label of x
     * @param x
     * @param label
     * @return the nearest distance in the label
     */
    double Get_Label(double *x, int *label);
    /**
     *
     * @param x
     * @param u
     * @param Dim_NUM
     * @return the distance between x and the label
     */
    double Compute_Distance(double *x, double *u, int Dim_NUM);
};


#endif //GRAB_CUT_KMEANSSAMPLE_H
