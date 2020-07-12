//+
Point(1) = {1, 0, 0, 1.0};
//+
Point(2) = {10, 0, 0, 1.0};
//+
Point(3) = {10, 10, 0, 1.0};
//+
Point(4) = {0, 10, 0, 1.0};
//+
Point(5) = {0, 1, 0, 1.0};
//+
Point(6) = {0, 0.8, 0, 1.0};
//+
Point(7) = {0, 0.6, 0, 1.0};
//+
Point(8) = {0, 0.5, 0, 1.0};
//+
Point(9) = {0, -10, 0, 1.0};
//+
Point(10) = {1, -10, 0, 1.0};
//+
Point(11) = {1, -0.2, 0, 1.0};
//+
Point(12) = {1, -0.4, 0, 1.0};
//+
Point(13) = {1, -0.6, 0, 1.0};
//+
Point(14) = {1, -0.8, 0, 1.0};
//+
Point(15) = {1, -1, 0, 1.0};
//+
Point(16) = {1.2, 0, 0, 1.0};
//+
Point(17) = {1.4, 0, 0, 1.0};
//+
Point(18) = {1.6, 0, 0, 1.0};
//+
Point(19) = {1.8, 0, 0, 1.0};
//+
Point(20) = {2, 0, 0, 1.0};
//+
Point(21) = {1, 0, 0, 1.0};
//+
Point(22) = {0, 0.5-4.67e-2, 0, 1.0};
//+
Point(23) = {0.02335, 0.5-4.67e-2+0.04044338636, 0, 1.0};
//+
Line(1) = {21, 16};
//+
Line(2) = {16, 17};
//+
Line(3) = {17, 18};
//+
Line(4) = {18, 19};
//+
Line(5) = {19, 20};
//+
Line(6) = {20, 2};
//+
Line(7) = {2, 3};
//+
Line(8) = {3, 4};
//+
Line(9) = {4, 5};
//+
Line(10) = {5, 6};
//+
Line(11) = {6, 7};
//+
Line(12) = {7, 8};
//+
Line(13) = {8, 22};
//+
Line(14) = {22, 9};
//+
Line(15) = {9, 10};
//+
Line(16) = {10, 15};
//+
Line(17) = {15, 14};
//+
Line(18) = {14, 13};
//+
Line(19) = {13, 12};
//+
Line(20) = {12, 11};
//+
Line(21) = {11, 21};
//+
Circle(22) = {8, 22, 23};
//+
Line(23) = {23, 21};
//+
Curve Loop(1) = {6, 7, 8, 9, 10, 11, 12, 22, 23, 1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {14, 15, 16, 17, 18, 19, 20, 21, -23, -22, 13};
//+
Plane Surface(2) = {2};
//+
Physical Curve("Bottom_Wall") = {1, 2, 3, 4, 5, 6};
//+
Physical Curve("Lateral_Wall_R") = {7};
//+
Physical Curve("Top_Wall") = {8};
//+
Physical Curve("Lateral_Wall_L") = {9, 10, 11, 12};
//+
Physical Curve("Tube_Wall_L") = {13, 14};
//+
Physical Curve("Inlet") = {15};
//+
Physical Curve("Tube_Wall_R") = {16, 17, 18, 19, 20, 21};
//+
Physical Curve("Meniscus") = {23, 22};
//+
Physical Surface("Vacuum") = {1};
//+
Physical Surface("Liquid") = {2};
//+
Transfinite Curve {1} = 100 Using Progression 1;
//+
Transfinite Curve {2} = 100 Using Progression 1;
//+
Transfinite Curve {3} = 100 Using Progression 1;
//+
Transfinite Curve {4} = 100 Using Progression 1;
//+
Transfinite Curve {5} = 100 Using Progression 1;
//+
Transfinite Curve {21} = 200 Using Progression 1;
//+
Transfinite Curve {20} = 100 Using Progression 1;
//+
Transfinite Curve {19} = 100 Using Progression 1;
//+
Transfinite Curve {18} = 100 Using Progression 1;
//+
Transfinite Curve {17} = 100 Using Progression 1;
//+
Transfinite Curve {16} = 200 Using Progression 1;
//+
Transfinite Curve {15} = 200 Using Progression 1;
//+
Transfinite Curve {14} = 300 Using Progression 1;
//+
Transfinite Curve {13} = 100 Using Progression 1;
//+
Transfinite Curve {6} = 100 Using Progression 1;
//+
Transfinite Curve {7} = 100 Using Progression 1;
//+
Transfinite Curve {8} = 100 Using Progression 1;
//+
Transfinite Curve {9} = 200 Using Progression 1;
//+
Transfinite Curve {10} = 100 Using Progression 1;
//+
Transfinite Curve {11} = 100 Using Progression 1;
//+
Transfinite Curve {12} = 200 Using Progression 1;
//+
Transfinite Curve {19} = 100 Using Progression 1;
//+
Transfinite Curve {22} = 100 Using Progression 1;
//+
Transfinite Curve {23} = 800 Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Surface{2};
