/*
Copyright 2022 Adobe. All rights reserved.
This file is licensed to you under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
OF ANY KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.
*/

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

//prasarna
using namespace Eigen;
using namespace std;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;


struct EllipseFittingData
{
	std::vector<Vector2d> points;
	std::vector<Vector2d> mTemp;
};

class GeometryUtils
{
public:

	static void computeEigenVectors(MatrixXd A, VectorXd &eValues, MatrixXd &eVectors, int dim)
	{
		EigenSolver<MatrixXd> es;

		es.compute(A);

		for (int e = 0; e<dim; e++)
		{
			eValues[e] = es.eigenvalues().real()[e];
			for (int f = 0; f<dim; f++)
			{
				eVectors(f, e) = es.eigenvectors().real().col(e)[f];
			}
		}
	}

	static int solveLeastSquares(SpMat A, SpMat B, SpMat &X)
	{
		Eigen::SimplicialCholesky<SpMat> chol(A); // performs a Cholesky factorization of A
		X = chol.solve(B); // use the factorization to solve for the given right hand side

		return 0;
	}

	static int solveLeastSquares(SpMat A, VectorXd b, VectorXd &x)
	{
		Eigen::SimplicialCholesky<SpMat> chol(A); // performs a Cholesky factorization of A
		x = chol.solve(b); // use the factorization to solve for the given right hand side

		return 0;
	};

	static int solveLeastSquares(MatrixXd A, VectorXd b, VectorXd &x)
	{
		MatrixXd ATA(A.cols(), A.cols());
		ATA.noalias() = A.transpose() * A;

		Eigen::FullPivLU<MatrixXd> lu(ATA);
		if (!lu.isInvertible())
		{
			printf("Matrix is not invertible!\n");
			return 1;
		}

		x = ATA.inverse() * A.transpose() * b;
		return 0;
	}

	static double SqrDistanceSpecial(const Vector2d e, const Vector2d y, Vector2d &x)
	{
		double sqrDistance;
		if (y[1] > 0.0)
		{
			if (y[0] > 0.0)
			{
				// Bisect to compute the root of F(t) for t >= -e1*e1.
				double esqr[2] = { e[0] * e[0], e[1] * e[1] };
				double ey[2] = { e[0] * y[0], e[1] * y[1] };
				double t0 = -esqr[1] + ey[1];
				double t1 = -esqr[1] + sqrt(ey[0] * ey[0] + ey[1] * ey[1]);
				double t = t0;
				const int imax = 2 * std::numeric_limits<double>::max_exponent;
				for (int i = 0; i < imax; ++i)
				{
					t = 0.5*(t0 + t1);
					if (t == t0 || t == t1)
					{
						break;
					}

					double r[2] = { ey[0] / (t + esqr[0]), ey[1] / (t + esqr[1]) };
					double f = r[0] * r[0] + r[1] * r[1] - 1.0;
					if (f > 0.0)
					{
						t0 = t;
					}
					else if (f < 0.0)
					{
						t1 = t;
					}
					else
					{
						break;
					}
				}

				x[0] = esqr[0] * y[0] / (t + esqr[0]);
				x[1] = esqr[1] * y[1] / (t + esqr[1]);
				double d[2] = { x[0] - y[0], x[1] - y[1] };
				sqrDistance = d[0] * d[0] + d[1] * d[1];
			}
			else  // y0 == 0
			{
				x[0] = 0.0;
				x[1] = e[1];
				double diff = y[1] - e[1];
				sqrDistance = diff * diff;
			}
		}
		else  // y1 == 0
		{
			double denom0 = e[0] * e[0] - e[1] * e[1];
			double e0y0 = e[0] * y[0];
			if (e0y0 < denom0)
			{
				// y0 is inside the subinterval.
				double x0de0 = e0y0 / denom0;
				double x0de0sqr = x0de0 * x0de0;
				x[0] = e[0] * x0de0;
				x[1] = e[1] * sqrt(fabs(1.0 - x0de0sqr));
				double d0 = x[0] - y[0];
				sqrDistance = d0 * d0 + x[1] * x[1];
			}
			else
			{
				// y0 is outside the subinterval.  The closest ellipse point has
				// x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
				x[0] = e[0];
				x[1] = 0.0;
				double diff = y[0] - e[0];
				sqrDistance = diff * diff;
			}
		}
		return sqrDistance;
	}

	static double SqrDistance(const Vector2d e, const Vector2d y, Vector2d &x)
	{
		// Determine reflections for y to the first quadrant.
		bool reflect[2];
		int i, j;
		for (i = 0; i < 2; ++i)
		{
			reflect[i] = (y[i] < 0.0);
		}

		// Determine the axis order for decreasing extents.
		int permute[2];
		if (e[0] < e[1])
		{
			permute[0] = 1;  permute[1] = 0;
		}
		else
		{
			permute[0] = 0;  permute[1] = 1;
		}

		int invpermute[2];
		for (i = 0; i < 2; ++i)
		{
			invpermute[permute[i]] = i;
		}

		Vector2d locE, locY;
		for (i = 0; i < 2; ++i)
		{
			j = permute[i];
			locE[i] = e[j];
			locY[i] = y[j];
			if (reflect[j])
			{
				locY[i] = -locY[i];
			}
		}

		Vector2d locX;
		double sqrDistance = SqrDistanceSpecial(locE, locY, locX);

		// Restore the axis order and reflections.
		for (i = 0; i < 2; ++i)
		{
			j = invpermute[i];
			if (reflect[i])
			{
				locX[j] = -locX[j];
			}
			x[i] = locX[j];
		}

		return sqrDistance;
	}

	static void Ellipse2DInitialGuess(std::vector<Vector2d> &points, Vector2d &center, double &angle, std::vector<double> &diag)
	{
		int noPoints = points.size();

		MatrixXd B(noPoints, 6);
		for (int i = 0; i<noPoints; i++)
		{
			B(i, 0) = points[i](0);
			B(i, 1) = points[i](1);
			B(i, 2) = 1.0;
			B(i, 3) = points[i](0)*points[i](0);
			B(i, 4) = points[i](0)*points[i](1)*sqrt(2.0);
			B(i, 5) = points[i](1)*points[i](1);
		}
		//qr decomposition
		HouseholderQR<MatrixXd> qr(B);
		MatrixXd R = qr.matrixQR().triangularView<Upper>();
		MatrixXd Q = qr.householderQ();
		MatrixXd R11(3, 3);
		MatrixXd R12(3, 3);
		MatrixXd R22(3, 3);
		for (int i = 0; i<3; i++)
		{
			for (int j = 0; j<3; j++)
			{
				R11(i, j) = R(i, j);
				R12(i, j) = R(i, j + 3);
				R22(i, j) = R(i + 3, j + 3);
			}
		}
		JacobiSVD<MatrixXd> svdofR(R22, Eigen::ComputeFullV);
		MatrixXd V = svdofR.matrixV();

		Vector3d w = V.col(2);

		VectorXd b(3);
		b = R12 * w;
		VectorXd v(3);
		solveLeastSquares(R11, b, v);
		v *= -1;
		MatrixXd A(2, 2);
		A.setZero();
		A(0, 0) = w(0);
		A(0, 1) = (1.0 / sqrt(2.0)) * w(1);
		A(1, 0) = A(0, 1);
		A(1, 1) = w(2);
		Vector2d bv;
		bv(0) = v(0); bv(1) = v(1);
		double c = v(2);
		VectorXd eigenValues(2);
		MatrixXd eigenVectors(2, 2);
		computeEigenVectors(A, eigenValues, eigenVectors, 2);
		Vector2d ev = eigenValues.real();
		Matrix2d eVectors = eigenVectors.real();
		eVectors.transposeInPlace();

		if (ev(0) * ev(1) <= 0.0)
		{
			printf("Error: Linear fit did not produce an ellipse!\n");
			return;
		}

		VectorXd t(2);
		solveLeastSquares(A, bv, t);
		t *= -0.5;
		double c_h = (t.transpose() * A * t);
		c_h = c_h + (bv.transpose()*t);
		c_h += c;

		center = t;
		diag[0] = sqrt(-c_h / ev(0));
		diag[1] = sqrt(-c_h / ev(1));

		angle = atan2(eVectors(0, 1), eVectors(0, 0));
	}

	static bool fillEllipseFunctionJacobian(std::vector<Vector2d> &points, VectorXd &u, VectorXd &f, SpMat &J)
	{
		//Define the system of nonlinear equations and Jacobian.

		//Tolerance for whether it is a circle
		double circleTol = 1e-5;

		int noPoints = points.size();

		VectorXd phi(noPoints);
		for (int i = 0; i<noPoints; i++)
			phi(i) = u(i);
		double alpha = u(noPoints);
		double a = u(noPoints + 1);
		double b = u(noPoints + 2);
		Vector2d z;
		z(0) = u(noPoints + 3);
		z(1) = u(noPoints + 4);

		//If it is a circle, the Jacobian will be singular, and the
		//Gauss-Newton step won't work.
		if (fabs(a - b) / (a + b) < circleTol)
		{
			printf("Ellipse is near circular!\n");
			return false;
		}

		VectorXd c(noPoints);
		VectorXd s(noPoints);
		for (int i = 0; i<noPoints; i++)
		{
			c(i) = cos(phi(i));
			s(i) = sin(phi(i));
		}
		double ca = cos(alpha);
		double sa = sin(alpha);

		Matrix2d Q, Qdot;
		Q(0, 0) = ca; Q(0, 1) = -sa;
		Q(1, 0) = sa; Q(1, 1) = ca;

		Qdot(0, 0) = -sa; Qdot(0, 1) = -ca;
		Qdot(1, 0) = ca; Qdot(1, 1) = -sa;

		//Preallocate function and Jacobian variables
		f.setZero();
		J.setZero();

		vector<T> coefficients;

		for (int i = 0; i<noPoints; i++)
		{
			Vector2d pt = points[i];
			Vector2d tmp(a*c(i), b*s(i));
			tmp = Q * tmp;
			pt = pt - z - tmp;
			f(i * 2) = pt(0);
			f(i * 2 + 1) = pt(1);

			tmp(0) = -a * s(i);
			tmp(1) = b * c(i);
			tmp = -Q * tmp;
			coefficients.push_back(T(i * 2, i, tmp(0)));
			coefficients.push_back(T(i * 2 + 1, i, tmp(1)));

			tmp(0) = a * c(i);
			tmp(1) = b * s(i);
			tmp = -Qdot * tmp;
			coefficients.push_back(T(i * 2, noPoints, tmp(0)));
			coefficients.push_back(T(i * 2 + 1, noPoints, tmp(1)));

			tmp(0) = c(i);
			tmp(1) = 0.0;
			tmp = -Q * tmp;
			coefficients.push_back(T(i * 2, noPoints + 1, tmp(0)));
			coefficients.push_back(T(i * 2 + 1, noPoints + 1, tmp(1)));

			tmp(0) = 0.0;
			tmp(1) = s(i);
			tmp = -Q * tmp;
			coefficients.push_back(T(i * 2, noPoints + 2, tmp(0)));
			coefficients.push_back(T(i * 2 + 1, noPoints + 2, tmp(1)));

			coefficients.push_back(T(i * 2, noPoints + 3, -1.0));
			coefficients.push_back(T(i * 2 + 1, noPoints + 3, 0.0));

			coefficients.push_back(T(i * 2, noPoints + 4, 0.0));
			coefficients.push_back(T(i * 2 + 1, noPoints + 4, -1.0));

		}

		J.setFromTriplets(coefficients.begin(), coefficients.end());

		return true;
	}

	static bool EllipseNonLinearFit(std::vector<Vector2d> &points, Vector2d &center, double &angle, std::vector<double> &diag)
	{
		// Gauss-Newton least squares ellipse fit minimising geometric distance
		int noPoints = points.size();
		MatrixXd M(2, noPoints);
		for (int i = 0; i<noPoints; i++)
		{
			M(0, i) = points[i](0) - center(0);
			M(1, i) = points[i](1) - center(1);
		}

		Matrix2d rotate;
		rotate(0, 0) = cos(angle); rotate(0, 1) = -sin(angle);
		rotate(1, 0) = sin(angle); rotate(1, 1) = cos(angle);

		MatrixXd R(2, noPoints);
		R = rotate.transpose() * M;

		//get initial phase estimates
		VectorXd phi0(noPoints);
		for (int i = 0; i<noPoints; i++)
		{
			phi0(i) = atan2(R(1, i), R(0, i));
		}

		//cout << "phi0:\n" << phi0 << endl;

		VectorXd u(noPoints + 5);
		for (int i = 0; i<noPoints; i++)
			u(i) = phi0(i);
		u(noPoints) = angle;
		u(noPoints + 1) = diag[0];
		u(noPoints + 2) = diag[1];
		u(noPoints + 3) = center(0);
		u(noPoints + 4) = center(1);

		//Iterate using Gauss Newton
		bool fConverged = false;
		int maxits = 200;
		double tol = 1e-5;

		VectorXd f(2 * noPoints);
		SpMat J(2 * noPoints, noPoints + 5);
		SpMat JTJ(noPoints + 5, noPoints + 5);
		VectorXd JTf(noPoints + 5);

		for (int nIts = 0; nIts<maxits; nIts++)
		{
			if (!fillEllipseFunctionJacobian(points, u, f, J))
			{
				fConverged = false;
				break;
			}

			JTJ = J.transpose() * J;
			JTf = J.transpose() * f;

			VectorXd h(noPoints + 5);
			solveLeastSquares(JTJ, JTf, h);
			h *= -1.0;

			u = u + h;

			//check for convergence
			double delta = h.norm() / u.norm();
			if (delta < tol)
			{
				fConverged = true;
				break;
			}
		}

		angle = u(noPoints);
		diag[0] = u(noPoints + 1);
		diag[1] = u(noPoints + 2);
		center(0) = u(noPoints + 3);
		center(1) = u(noPoints + 4);

		return fConverged;
	}

	static double EllipseFit2D(std::vector<Vector2d> &points, Vector2d &center, Matrix2d &rotate, std::vector<double> &diag)
	{
		//find the center
		Vector2d centroid(0.0, 0.0);
		for (int i = 0; i<points.size(); i++)
		{
			centroid += points[i];
		}
		centroid /= (double)(points.size());

		vector<Vector2d> tmpPoints;
		for (int i = 0; i<points.size(); i++)
		{
			tmpPoints.push_back(points[i] - centroid);
		}

		//get initial estimate
		Vector2d initialCenter;
		double initialAngle;
		vector<double> initialDiag;
		initialDiag.resize(2);
		Ellipse2DInitialGuess(tmpPoints, initialCenter, initialAngle, initialDiag);

		//minimize error
		center = initialCenter;
		double angle = initialAngle;
		diag[0] = initialDiag[0];
		diag[1] = initialDiag[1];
		bool converged = EllipseNonLinearFit(tmpPoints, center, angle, diag);
		if (!converged)
		{
			center = initialCenter;
			angle = initialAngle;
			diag[0] = initialDiag[0];
			diag[1] = initialDiag[1];
		}

		rotate(0, 0) = cos(angle); rotate(0, 1) = -sin(angle);
		rotate(1, 0) = sin(angle); rotate(1, 1) = cos(angle);

		center += centroid;

		double err = 0.0;
		for (int i = 0; i<points.size(); i++)
		{
			err += computeDistanceFromPointToEllipse2D(points[i], center, rotate, diag);
		}

		return err;
	}

	static double computeDistanceFromPointToEllipse2D(Vector2d point, Vector2d &center, Matrix2d &rotate, std::vector<double> &diag)
	{
		Vector2d diff(point(0) - center[0],
			point(1) - center[1]);

		Vector2d p;
		p[0] = diff(0)*rotate(0, 0) + diff(1)*rotate(1, 0);
		p[1] = diff(0)*rotate(0, 1) + diff(1)*rotate(1, 1);
		Vector2d x;

		double sqrDistance = SqrDistance(Vector2d(diag[0], diag[1]), p, x);

		return sqrt(sqrDistance);
	}



};

