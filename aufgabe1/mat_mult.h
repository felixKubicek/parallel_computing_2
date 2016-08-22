
#define M(m, row, col) m.elements[(row) * m.cols + (col)]
#define SIZE(m) m.cols * m.rows * sizeof(m_cell)

#define m_cell_fms "%d"
typedef int m_cell;

typedef struct { 
    int rows;
    int cols; 
    m_cell* elements;
} Matrix;

