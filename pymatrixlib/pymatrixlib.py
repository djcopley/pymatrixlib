import numpy


class MatrixError(Exception):
    pass


class Matrix:
    def __init__(self, *args, matrix=None, rows=None, cols=None, dtype=None):
        """
        Matrix Constructor
        -----------------
        If 'matrix' is passed, __init__ will act as a copy constructor. Alternatively, if 'rows' and 'cols' are
        specified, __init__ constructs a new zero matrix. Note: argument 'matrix' is mutually exclusive with 'rows',
        'cols', and 'dtype'. Do not pass both.

        :param Matrix or numpy.ndarray matrix: object to copy

        :param int rows: number of rows
        :param int cols: number of columns
        :param dtype: numpy data type (optional)

        Examples
        --------
        >>> Matrix(3, 3)
        ┌ 0.0 0.0 0.0 ┐
        | 0.0 0.0 0.0 |
        └ 0.0 0.0 0.0 ┘

        >>> Matrix(3, 3, numpy.int)
        ┌ 0 0 0 ┐
        | 0 0 0 |
        └ 0 0 0 ┘

        >>> A = Matrix(3, 3)
        >>> A[0][1] = A[1][1] = A[2][1] = 25
        >>> Matrix(A)
        ┌  0.0 25.0  0.0 ┐
        |  0.0 25.0  0.0 |
        └  0.0 25.0  0.0 ┘

        >>> Matrix(rows=3, cols=3, dtype=int)
        ┌ 0 0 0 ┐
        | 0 0 0 |
        └ 0 0 0 ┘
        """

        if len(args) == 1:
            matrix = args[0]

        elif len(args) >= 2:
            rows = args[0]
            cols = args[1]

            if len(args) >= 3:
                dtype = args[2]

        if not (matrix is not None or (rows and cols)):
            raise TypeError("{}() missing required argument(s): 'matrix' or ('rows' and 'cols')".format(
                self.__class__.__name__))
        elif matrix is not None and not (isinstance(matrix, Matrix) or isinstance(matrix, numpy.ndarray)):
            raise TypeError("argument 'matrix' must be type 'Matrix' or 'numpy.ndarray'")
        elif not rows and cols and (isinstance(rows, int) or isinstance(cols, int)):
            raise TypeError("argument(s) 'rows' and 'cols' must be type 'int'")

        # Copy matrix values into self
        if matrix is not None:
            if isinstance(matrix, Matrix):
                self._matrix = matrix._matrix.copy()
                self._rows = matrix._rows
                self._cols = matrix._cols
            elif isinstance(matrix, numpy.ndarray):
                self._matrix = matrix.copy()
                self._rows = len(matrix)
                self._cols = len(matrix[0])

        # Construct new matrix
        elif rows and cols:
            self._rows = rows
            self._cols = cols
            self._matrix = numpy.zeros((rows, cols), dtype=dtype if dtype else numpy.double)

    def __str__(self):
        """
        Method returns the string representation of the matrix

        :return str: string representation of the matrix
        """
        # The length of the longest matrix element
        max_len_col = numpy.zeros((self.cols,), dtype=numpy.int)

        # Number of additional spaces to pad matrix elements
        padding = 2

        # String representation of the matrix
        str_repr = ""

        # Get the length of the longest element in each col (to help formatting)
        for col_idx in range(self.cols):
            for row_idx in range(self.rows):
                max_len_col[col_idx] = max(len(str(self[row_idx][col_idx])), max_len_col[col_idx])

        for row_idx in range(self.rows):

            if row_idx == 0:
                str_repr += "┌"
            elif row_idx == self.rows - 1:
                str_repr += "└"
            else:
                str_repr += "|"

            for col_idx in range(self.cols):
                str_repr += " " * (max_len_col[col_idx] + padding - len(str(self[row_idx][col_idx]))) + str(
                    self[row_idx][col_idx])

            str_repr += " " * padding
            if row_idx == 0:
                str_repr += "┐"
            elif row_idx == self.rows - 1:
                str_repr += "┘"
            else:
                str_repr += "|"
            str_repr += "\n"

        return str(str_repr)

    def __repr__(self):
        """
        Method returns the string representation of the matrix

        :return str: string representation of the matrix
        """
        return str(self)

    def __getitem__(self, item):
        return self._matrix[item]

    def __setitem__(self, key, value):
        self._matrix[key] = value

    def __copy__(self):
        copy = Matrix(self.rows, self.cols)
        copy._matrix = self._matrix
        return copy

    def __deepcopy__(self, memodict):
        dc = Matrix(self.rows, self.cols)
        memodict[id(self)] = dc
        dc._matrix = self._matrix.copy()
        return dc

    def __add__(self, other):
        """
        Add two matrices

        :param other:
        :return:
        """
        if not isinstance(other, Matrix):
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))

        if self.rows != other.rows or self.cols != other.cols:
            raise MatrixError("cannot add matrix of dimensions ({}, {}) and ({}, {})".format(
                self.rows, self.cols, other.rows, other.cols))

        res = Matrix(self.rows, self.cols)

        for row_idx in range(res.rows):
            for col_idx in range(res.cols):
                res[row_idx][col_idx] = self[row_idx][col_idx] + other[row_idx][col_idx]

        return res

    def __sub__(self, other):
        """
        Subtract two matrices

        :param other:
        :return:
        """
        if not isinstance(other, Matrix):
            raise TypeError("unsupported operand type(s) for -: '{}' and '{}'".format(type(self), type(other)))

        if self.rows != other.rows or self.cols != other.cols:
            raise MatrixError("cannot subtract matrix of dimensions ({}, {}) and ({}, {})".format(
                self.rows, self.cols, other.rows, other.cols))

        res = Matrix(self.rows, self.cols)

        for row_idx in range(res.rows):
            for col_idx in range(res.cols):
                res[row_idx][col_idx] = self[row_idx][col_idx] - other[row_idx][col_idx]

        return res

    def __mul__(self, other):
        """
        Multiply two matrices

        :param [Matrix, int] other:
        :return:
        """
        # Scalar multiplication
        if isinstance(other, int):
            res = Matrix(self.rows, self.cols)
            for row_idx in range(self.rows):
                for col_idx in range(self.cols):
                    res[row_idx][col_idx] = other * self[row_idx][col_idx]

        if not isinstance(other, Matrix):
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

        # Matrix multiplication
        if self.cols != other.rows:
            raise MatrixError("cannot multiply matrix of dimensions ({}, {}) and ({}, {})".format(
                self.rows, self.cols, other.rows, other.cols))

        res = Matrix(self.cols, other.rows)

        for row_idx in range(res.rows):
            for col_idx in range(res.cols):
                for n in range(self.cols):
                    res[row_idx][col_idx] += self[row_idx][n] * other[n][col_idx]

        return res

    def __truediv__(self, other):
        """

        :param other:
        :return:
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

        res = Matrix(self)

        for row_idx in range(res.rows):
            for col_idx in range(res.cols):
                res[row_idx][col_idx] /= other

        return res

    def __floordiv__(self, other):
        """

        :param other:
        :return:
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise TypeError("unsupported operand type(s) for /: '{}' and '{}'".format(type(self), type(other)))

        res = Matrix(self)

        for row_idx in range(res.rows):
            for col_idx in range(res.cols):
                res[row_idx][col_idx] //= other

        return res

    def __pow__(self, power):
        """
        Method performs exponentiation on the matrix

        :param int power: exponent to be raised to
        :return Matrix: matrix
        """
        if not self.square:
            raise MatrixError("exponentiation can only be applied to a square matrix")

        if power < 0:
            # Calculate the inverse matrix
            inv = self.inv()

            # Return the inverse to the positive power
            return inv ** abs(power)

        elif power == 0:
            return IdentityMatrix(self.rows, self.cols)

        # For any positive power
        res = self

        # Iterate and multiply over range 1 to power
        for p in range(1, power):
            res *= self

        return res

    def __eq__(self, other):
        """
        Equal operator

        :param Matrix other: matrix to test equality against
        :return bool: equality of the matrices
        """
        if not isinstance(other, Matrix) or self.rows != other.rows or self.cols != other.cols:
            return False

        # Iterate and check equality of each element
        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                if self[row_idx][col_idx] != other[row_idx][col_idx]:
                    return False

        return True

    def __ne__(self, other):
        """
        Not equal operator

        :param Matrix other: matrix to test non-equality against
        :return bool: non-equality of the matrices
        """
        return not self == other

    def __hash__(self):
        """
        Method computes and returns the hash of the Matrix data array

        :return: hash of matrix
        """
        return hash(self._matrix)

    @property
    def rows(self):
        """
        Getter method for the number of rows

        :return int: number of rows
        """
        return self._rows

    @property
    def cols(self):
        """
        Getter method for the number of columns

        :return int: number of columns
        """
        return self._cols

    @property
    def size(self):
        """
        Method returns the size of the matrix

        :return tuple: size
        """
        return self.rows, self.cols

    @property
    def square(self):
        """
        Method returns true it is a square matrix

        :return bool: True if square matrix else False
        """
        return self.rows == self.cols

    @property
    def rank(self):
        """
        :return int:
        """
        # Todo
        return

    @property
    def nullity(self):
        """
        :return int:
        """
        # Todo
        return

    def sub_matrix(self, row, col):
        """
        Method returns a sub-matrix with the specified row and col removed

        :param int row: row to be removed from sub-matrix
        :param int col: col to be removed from sub-matrix
        :return Matrix: sub-matrix with row and col removed
        """
        sub_matrix = Matrix(self.rows - 1, self.cols - 1)
        # row and column counters for sub_matrix
        r = c = 0
        for row_idx in range(self.rows):
            if row_idx != row:
                for col_idx in range(self.cols):
                    if col_idx != col:
                        sub_matrix[r][c] = self[row_idx][col_idx]
                        c += 1
                c = 0
                r += 1
        return sub_matrix

    def transpose(self):
        """
        Method calculates and returns the transpose matrix

        :return Matrix: transpose matrix
        """
        a_t = Matrix(self.cols, self.rows)
        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                a_t[col_idx][row_idx] = self[row_idx][col_idx]

        return a_t

    def det(self):
        """
        Method calculates and returns the determinant of the matrix

        :return Matrix: determinant
        """
        if not self.square:
            raise MatrixError("determinant can only be computed for a square matrix")

        # recursive determinant calculator
        def determinant(matrix):
            # 2x2 matrix
            if 2 == matrix.rows == matrix.cols:
                """
                det( ┌ a b ┐
                     └ c d ┘ ) = ad - bc
                """
                return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

            else:
                # Result
                res = 0
                # Get the top row
                for index, item in enumerate(matrix[0]):
                    # Drop top row and current col from sub matrix
                    # Alternating sign for determinant
                    res += ((-1) ** index) * item * determinant(matrix.sub_matrix(0, index))
                return res

        return determinant(self)

    def inv(self):
        """
        Method calculates and returns the inverse matrix

        :return Matrix: inverse matrix
        """
        if not self.square:
            raise MatrixError("cannot calculate the inverse of a non-square matrix")

        # Augment matrix with identity matrix for the gauss-jordan elimination inverse method
        reduced_matrix = AugmentedMatrix(self, IdentityMatrix(self.rows, self.cols)).rref()

        # Return a non-augmented matrix
        return Matrix(reduced_matrix)  # Copy constructor will copy matrix into new array

    def ref(self):
        """
        Row Echelon Form

        Method calculates and returns the row echelon form of the matrix.

        :return Matrix: row echelon form
        """
        # Todo
        pass

    def rref(self):
        """
        Reduced Row Echelon Form

        Method calculates and returns the reduced row echelon form of the matrix.

        :return Matrix: reduced row echelon form
        """
        reduced_matrix = Matrix(self)

        lead = 0

        for row_idx in range(reduced_matrix.rows):
            if reduced_matrix.cols <= lead:
                break
            pointer = row_idx
            while reduced_matrix[pointer, lead] == 0:
                pointer += 1
                if reduced_matrix.rows == pointer:
                    pointer = row_idx
                    lead += 1
                    if reduced_matrix.cols == lead:
                        break

            # Swap rows row_idx and i
            temp = reduced_matrix[row_idx]
            reduced_matrix[row_idx] = reduced_matrix[pointer]
            reduced_matrix[pointer] = temp

            if reduced_matrix[row_idx][lead] != 0:
                reduced_matrix[row_idx] /= reduced_matrix[row_idx, lead]

            for pointer in range(reduced_matrix.rows):
                if pointer != row_idx:
                    reduced_matrix[pointer] -= reduced_matrix[pointer][lead] * reduced_matrix[row_idx]

            lead += 1

        return reduced_matrix


class IdentityMatrix(Matrix):
    def __init__(self, rows, cols, dtype=None):
        super().__init__(rows, cols, dtype=dtype)

        for row_idx in range(min(self.rows, self.cols)):
            self[row_idx][row_idx] = 1


class OnesMatrix(Matrix):
    def __init__(self, rows, cols, dtype=None):
        super().__init__(numpy.ones((rows, cols), dtype=dtype))


class AugmentedMatrix(Matrix):
    def __init__(self, matrix, b):
        """
        AugmentedMatrix Constructor
        ---------------------------

        :param Matrix or numpy.ndarray matrix:
        :param Matrix b: augmented matrix / vector
        """
        super().__init__(matrix)
        self._b_matrix = b

    def ref(self):
        return super().ref()

    def rref(self):
        return super().rref()
